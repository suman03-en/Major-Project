import tempfile
import unittest
from pathlib import Path

from src.agent.session_store import SQLiteSessionStore

try:
    from fastapi.testclient import TestClient

    import web.main as web_main

    _HAS_FASTAPI = True
except ModuleNotFoundError:
    _HAS_FASTAPI = False


@unittest.skipUnless(_HAS_FASTAPI, "fastapi not installed in current test environment")
class TestConversationsAPI(unittest.TestCase):
    def test_conversations_crud_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteSessionStore(db_path=str(Path(tmpdir) / "api.db"))
            session = store.get_or_create("s-api-1")
            session.add_message("user", "कम्पनी दर्ता कसरी गर्ने?")
            session.add_message("assistant", "दफा ३ अनुसार दर्ता गर्नुपर्छ।")
            store.save(session)

            orig_store = web_main.session_store
            web_main.session_store = store
            try:
                client = TestClient(web_main.app, raise_server_exceptions=True)
                res = client.get("/conversations")
                self.assertEqual(res.status_code, 200)
                convs = res.json()["conversations"]
                self.assertEqual(len(convs), 1)
                self.assertEqual(convs[0]["session_id"], "s-api-1")

                detail = client.get("/conversations/s-api-1")
                self.assertEqual(detail.status_code, 200)
                self.assertEqual(len(detail.json()["messages"]), 2)

                renamed = client.patch("/conversations/s-api-1", json={"title": "My Chat"})
                self.assertEqual(renamed.status_code, 200)
                self.assertEqual(renamed.json()["title"], "My Chat")

                deleted = client.delete("/conversations/s-api-1")
                self.assertEqual(deleted.status_code, 200)
                self.assertEqual(client.get("/conversations/s-api-1").status_code, 404)
            finally:
                web_main.session_store = orig_store
                store.close()

    def test_index_and_static_assets_served(self) -> None:
        client = TestClient(web_main.app, raise_server_exceptions=True)
        index_res = client.get("/")
        self.assertEqual(index_res.status_code, 200)
        self.assertIn("/static/css/tokens.css", index_res.text)
        self.assertIn("/static/js/main.js", index_res.text)

        for asset_path in (
            "/static/css/tokens.css",
            "/static/css/layout.css",
            "/static/css/chat.css",
            "/static/css/composer.css",
            "/static/js/main.js",
            "/static/js/api.js",
            "/static/js/chat.js",
            "/static/js/sidebar.js",
            "/static/js/markdown.js",
        ):
            res = client.get(asset_path)
            self.assertEqual(res.status_code, 200, f"Missing static asset: {asset_path}")


class TestFrontendStructure(unittest.TestCase):
    def test_modular_css_js_and_partials_exist(self) -> None:
        web_dir = Path(__file__).resolve().parent.parent / "web"
        expected = [
            "templates/index.html",
            "templates/partials/sidebar.html",
            "templates/partials/header.html",
            "templates/partials/composer.html",
            "templates/partials/welcome.html",
            "static/css/tokens.css",
            "static/css/layout.css",
            "static/css/chat.css",
            "static/css/composer.css",
            "static/js/main.js",
            "static/js/api.js",
            "static/js/chat.js",
            "static/js/sidebar.js",
            "static/js/markdown.js",
        ]
        for rel in expected:
            self.assertTrue((web_dir / rel).is_file(), f"Expected file missing: {rel}")

        index_text = (web_dir / "templates/index.html").read_text(encoding="utf-8")
        self.assertNotIn("<style>", index_text)
        self.assertIn('/static/js/main.js', index_text)


if __name__ == "__main__":
    unittest.main()
