import unittest
from unittest.mock import AsyncMock

from api.research import ResearchRequest
from engine.research_workflow import collect_evidence, check_references
from engine.web_research import PublicPageCache, public_cache_seconds, source, ResearchError


class WorkflowTest(unittest.IsolatedAsyncioTestCase):
    async def test_off_never_calls_network_and_preserves_authorized_references(self):
        adapter = AsyncMock()
        document = {'kind': 'document_chunk', 'sourceId': 'owned', 'versionId': 'v1'}
        result = await collect_evidence(ResearchRequest(documentIds=['owned']), [document], adapter)
        self.assertEqual(result, [document])
        adapter.search.assert_not_called()
        self.assertEqual(await collect_evidence(ResearchRequest(), [], adapter), [])

    async def test_one_reviewed_query_and_bounded_interleaved_sources(self):
        documents = [{'kind': 'document_chunk', 'sourceId': str(i), 'versionId': 'v1'} for i in range(5)]
        web = [{'kind': 'web_page', 'sourceId': str(i)} for i in range(5)]
        adapter = AsyncMock()
        adapter.search.return_value = {'sources': web}
        request = ResearchRequest(mode='on', publicQuery='reviewed query', consentToExternalQuery=True, documentIds=['private'])
        result = await collect_evidence(request, documents, adapter)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[:2], [documents[0], web[0]])
        adapter.search.assert_awaited_once()
        sent = adapter.search.call_args.args[0].model_dump()
        self.assertEqual(sent['publicQuery'], 'reviewed query')
        self.assertEqual(sent['documentIds'], [])
        self.assertNotIn('private', str(sent))

    async def test_no_documents_fails_before_search_and_no_web_does_not_fallback(self):
        adapter = AsyncMock()
        request = ResearchRequest(mode='on', publicQuery='query', consentToExternalQuery=True, documentIds=['doc'])
        with self.assertRaises(ResearchError):
            await collect_evidence(request, [], adapter)
        adapter.search.assert_not_called()
        adapter.search.return_value = {'sources': []}
        with self.assertRaises(ResearchError):
            await collect_evidence(request.model_copy(update={'documentIds': []}), [], adapter)

    def test_reference_check_does_not_invent_support(self):
        self.assertEqual(check_references('Value [S1] and [S77]', [{'citationId': 'S1'}]),
                         'Value [S1] and [unverified reference]')
        self.assertEqual(check_references('No supported value is available.', []), 'No supported value is available.')


class PageCacheTest(unittest.TestCase):
    def test_cache_requires_explicit_public_freshness(self):
        self.assertEqual(public_cache_seconds({'cache-control': 'public, max-age=600'}), 60)
        self.assertEqual(public_cache_seconds({'cache-control': 'public, max-age=60', 'age': '50'}), 10)
        for headers in [{}, {'cache-control': 'max-age=600'},
                        {'cache-control': 'public, max-age=60', 'age': '70'},
                        {'cache-control': 'public, max-age=60', 'vary': 'Cookie'},
                        {'cache-control': 'public, max-age=60', 'set-cookie': 'a=b'},
                        {'cache-control': 'public, private, max-age=60'},
                        {'cache-control': 'public, no-cache, max-age=60'},
                        {'cache-control': 'public, max-age=60, max-age=1'},
                        {'cache-control': 'public, max-age=' + '9' * 5000},
                        {'cache-control': 'public, no-store, max-age=60'}]:
            self.assertEqual(public_cache_seconds(headers), 0)

    def test_expiry_capacity_timestamp_and_copies(self):
        now = [0]
        cache = PublicPageCache(2, lambda: now[0])
        def page(path):
            return source('https://example.com/' + path, 'Reference', 'public excerpt', 'web_page', b'public')
        first = page('a')
        cache.put(first, 60)
        copy = cache.get(str(first.url))
        self.assertEqual(copy.retrievedAt, first.retrievedAt)
        copy.excerpt = 'mutated'
        self.assertEqual(cache.get(str(first.url)).excerpt, 'public excerpt')
        now[0] = 59
        self.assertIsNotNone(cache.get(str(first.url)))
        now[0] = 60
        self.assertIsNone(cache.get(str(first.url)))
        for name in ('a', 'b', 'c'): cache.put(page(name), 60)
        self.assertEqual(len(cache.entries), 2)
        self.assertIsNone(cache.get(str(first.url)))
        cache.put(page('private?token=value'), 60)
        self.assertIsNone(cache.get('https://example.com/private?token=value'))
        with self.assertRaises(ResearchError): cache.get('https://127.0.0.1/')


if __name__ == '__main__': unittest.main()
