"""Obol: config-only (no database) model management.

/model/new, /model/delete, /model/update and PATCH /model/{id}/update must work
against the in-memory router and persist to config.yaml when the proxy runs
without a database (STORE_MODEL_IN_DB off), as the Obol Stack runs it.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, ModelInfoDelete, ProxyException, UserAPIKeyAuth
from litellm.proxy.management_endpoints.model_management_endpoints import (
    add_new_model,
    delete_model,
    patch_model,
    update_model,
)
from litellm.router import Router
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo, updateDeployment, updateLiteLLMParams

PAID_ROUTE = {
    "model_name": "paid/*",
    "litellm_params": {
        "model": "openai/*",
        "api_base": "http://x402-buyer.llm.svc.cluster.local:8402/v1",
        "api_key": "unused",
    },
    "model_info": {"id": "paid-route"},
}
EXISTING = {
    "model_name": "qwen",
    "litellm_params": {"model": "openai/qwen", "api_base": "http://ollama:11434/v1", "api_key": "k"},
    "model_info": {"id": "qwen-id"},
}
GENERAL_SETTINGS = {"master_key": "sk-obol-test"}

ADMIN = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
INTERNAL_USER = UserAPIKeyAuth(user_id="someone", user_role=LitellmUserRoles.INTERNAL_USER)


@dataclass(frozen=True)
class ConfigOnlyProxy:
    config_path: Path
    proxy_config: proxy_server.ProxyConfig
    router: Router

    def saved(self) -> dict:
        return yaml.safe_load(self.config_path.read_text())

    def saved_ids(self) -> list[str]:
        return [m["model_info"]["id"] for m in self.saved()["model_list"]]


@pytest.fixture
def config_only_proxy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ConfigOnlyProxy]:
    config = {
        "model_list": [PAID_ROUTE, EXISTING],
        "general_settings": GENERAL_SETTINGS,
        "litellm_settings": {"drop_params": True},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    proxy_config = proxy_server.ProxyConfig()
    proxy_config.update_config_state(config=config)
    router = Router(model_list=[PAID_ROUTE, EXISTING])

    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "store_model_in_db", False)
    monkeypatch.setattr(proxy_server, "proxy_config", proxy_config)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "user_config_file_path", str(config_path))
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "premium_user", False)
    yield ConfigOnlyProxy(config_path=config_path, proxy_config=proxy_config, router=router)


def _new_model(**model_info) -> Deployment:
    return Deployment(
        model_name="gpt-4o",
        litellm_params=LiteLLM_Params(model="openai/gpt-4o", api_key="sk-provider"),
        model_info=ModelInfo(**model_info),
    )


@pytest.mark.asyncio
async def test_add_serves_the_model_and_persists_it_with_its_id(config_only_proxy: ConfigOnlyProxy):
    result = await add_new_model(model_params=_new_model(id="new-id"), user_api_key_dict=ADMIN)

    assert result.model_id == "new-id"
    assert result.model_name == "gpt-4o"
    assert config_only_proxy.router.get_deployment(model_id="new-id") is not None
    saved = config_only_proxy.saved()
    assert config_only_proxy.saved_ids() == ["paid-route", "qwen-id", "new-id"]
    assert saved["model_list"][2]["litellm_params"]["model"] == "openai/gpt-4o"
    assert saved["general_settings"] == GENERAL_SETTINGS
    assert saved["litellm_settings"] == {"drop_params": True}
    assert config_only_proxy.proxy_config.get_config_state()["model_list"][2]["model_info"]["id"] == "new-id"


@pytest.mark.asyncio
async def test_add_generates_and_persists_an_id_when_none_is_given(config_only_proxy: ConfigOnlyProxy):
    result = await add_new_model(model_params=_new_model(), user_api_key_dict=ADMIN)

    assert result.model_id
    assert config_only_proxy.saved_ids()[-1] == result.model_id
    assert config_only_proxy.router.get_deployment(model_id=result.model_id) is not None


@pytest.mark.asyncio
async def test_add_keeps_the_paid_wildcard_route_serving(config_only_proxy: ConfigOnlyProxy):
    await add_new_model(model_params=_new_model(id="new-id"), user_api_key_dict=ADMIN)

    paid = config_only_proxy.router.get_deployment(model_id="paid-route")
    assert paid is not None
    assert paid.litellm_params.api_base == "http://x402-buyer.llm.svc.cluster.local:8402/v1"


@pytest.mark.asyncio
async def test_add_rejects_a_team_scoped_model(config_only_proxy: ConfigOnlyProxy):
    with pytest.raises(ProxyException) as exc_info:
        await add_new_model(model_params=_new_model(id="team-model", team_id="team-1"), user_api_key_dict=ADMIN)

    assert exc_info.value.code == "400"
    assert "Team-scoped" in str(exc_info.value.message)
    assert config_only_proxy.saved_ids() == ["paid-route", "qwen-id"]
    assert config_only_proxy.router.get_deployment(model_id="team-model") is None


@pytest.mark.asyncio
async def test_delete_stops_serving_and_unpersists_only_that_model(config_only_proxy: ConfigOnlyProxy):
    result = await delete_model(model_info=ModelInfoDelete(id="qwen-id"), user_api_key_dict=ADMIN)

    assert result == {"message": "Model: qwen-id deleted successfully"}
    assert config_only_proxy.router.get_deployment(model_id="qwen-id") is None
    assert config_only_proxy.saved_ids() == ["paid-route"]
    assert config_only_proxy.saved()["general_settings"] == GENERAL_SETTINGS


@pytest.mark.asyncio
async def test_delete_of_an_unknown_id_is_a_400(config_only_proxy: ConfigOnlyProxy):
    with pytest.raises(ProxyException) as exc_info:
        await delete_model(model_info=ModelInfoDelete(id="nope"), user_api_key_dict=ADMIN)

    assert exc_info.value.code == "400"
    assert "not found in config" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_update_merges_params_renames_and_reupserts(config_only_proxy: ConfigOnlyProxy):
    result = await update_model(
        model_params=updateDeployment(
            model_name="qwen-renamed",
            litellm_params=updateLiteLLMParams(model="openai/qwen2"),
            model_info=ModelInfo(id="qwen-id"),
        ),
        user_api_key_dict=ADMIN,
    )

    assert result["model_name"] == "qwen-renamed"
    assert result["model_info"]["id"] == "qwen-id"
    saved_entry = config_only_proxy.saved()["model_list"][1]
    assert saved_entry["litellm_params"] == {
        "model": "openai/qwen2",
        "api_base": "http://ollama:11434/v1",
        "api_key": "k",
    }
    live = config_only_proxy.router.get_deployment(model_id="qwen-id")
    assert live is not None
    assert live.model_name == "qwen-renamed"
    assert live.litellm_params.model == "openai/qwen2"


@pytest.mark.asyncio
async def test_update_of_an_unknown_id_is_a_400(config_only_proxy: ConfigOnlyProxy):
    with pytest.raises(ProxyException) as exc_info:
        await update_model(
            model_params=updateDeployment(
                litellm_params=updateLiteLLMParams(model="x"), model_info=ModelInfo(id="nope")
            ),
            user_api_key_dict=ADMIN,
        )

    assert exc_info.value.code == "400"


@pytest.mark.asyncio
async def test_patch_updates_by_path_id(config_only_proxy: ConfigOnlyProxy):
    result = await patch_model(
        model_id="qwen-id",
        patch_data=updateDeployment(litellm_params=updateLiteLLMParams(api_base="http://other:11434/v1")),
        user_api_key_dict=ADMIN,
    )

    assert result["model_info"]["id"] == "qwen-id"
    assert config_only_proxy.saved()["model_list"][1]["litellm_params"]["api_base"] == "http://other:11434/v1"
    live = config_only_proxy.router.get_deployment(model_id="qwen-id")
    assert live is not None
    assert live.litellm_params.api_base == "http://other:11434/v1"


@pytest.mark.asyncio
async def test_patch_of_an_unknown_id_is_a_404(config_only_proxy: ConfigOnlyProxy):
    with pytest.raises(ProxyException) as exc_info:
        await patch_model(
            model_id="nope",
            patch_data=updateDeployment(litellm_params=updateLiteLLMParams(api_base="http://x")),
            user_api_key_dict=ADMIN,
        )

    assert exc_info.value.code == "404"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [
        lambda: add_new_model(model_params=_new_model(id="new-id"), user_api_key_dict=INTERNAL_USER),
        lambda: delete_model(model_info=ModelInfoDelete(id="qwen-id"), user_api_key_dict=INTERNAL_USER),
        lambda: update_model(
            model_params=updateDeployment(model_name="x", model_info=ModelInfo(id="qwen-id")),
            user_api_key_dict=INTERNAL_USER,
        ),
        lambda: patch_model(
            model_id="qwen-id", patch_data=updateDeployment(model_name="x"), user_api_key_dict=INTERNAL_USER
        ),
    ],
    ids=["add", "delete", "update", "patch"],
)
async def test_non_admins_cannot_manage_models_without_a_db(config_only_proxy: ConfigOnlyProxy, call):
    original = config_only_proxy.config_path.read_text()

    with pytest.raises(ProxyException) as exc_info:
        await call()

    assert exc_info.value.code == "403"
    assert config_only_proxy.config_path.read_text() == original
    assert config_only_proxy.router.get_deployment(model_id="qwen-id").model_name == "qwen"


@pytest.mark.asyncio
async def test_connected_db_without_store_model_in_db_still_uses_config(
    config_only_proxy: ConfigOnlyProxy, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())

    await delete_model(model_info=ModelInfoDelete(id="qwen-id"), user_api_key_dict=ADMIN)

    assert config_only_proxy.saved_ids() == ["paid-route"]


@pytest.mark.asyncio
async def test_db_mode_keeps_upstream_db_path(config_only_proxy: ConfigOnlyProxy, monkeypatch: pytest.MonkeyPatch):
    prisma = MagicMock()
    prisma.db.litellm_proxymodeltable.find_unique = AsyncMock(return_value=None)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    original = config_only_proxy.config_path.read_text()

    with pytest.raises(ProxyException) as exc_info:
        await delete_model(model_info=ModelInfoDelete(id="qwen-id"), user_api_key_dict=ADMIN)

    assert "not found in db" in str(exc_info.value.message)
    prisma.db.litellm_proxymodeltable.find_unique.assert_awaited_once()
    assert config_only_proxy.config_path.read_text() == original
    assert config_only_proxy.router.get_deployment(model_id="qwen-id") is not None
