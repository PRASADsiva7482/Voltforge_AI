"""Bound bodies before JSON parsing, including chunked requests."""
from starlette.responses import JSONResponse


class RequestSizeLimit:
    def __init__(self, app, max_bytes):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        try:
            length = int(headers.get(b'content-length', b'0'))
        except ValueError:
            return await JSONResponse({'code': 'INVALID_CONTENT_LENGTH'}, status_code=400)(scope, receive, send)
        if length > self.max_bytes or length < 0:
            return await JSONResponse({'code': 'REQUEST_TOO_LARGE'}, status_code=413)(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            data = message.get('body', b'')
            size += len(data)
            if size > self.max_bytes:
                return await JSONResponse({'code': 'REQUEST_TOO_LARGE'}, status_code=413)(scope, receive, send)
            chunks.append(data)
            if not message.get('more_body', False):
                break
        delivered = False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': b''.join(chunks), 'more_body': False}
            return await receive()
        await self.app(scope, replay, send)
