"""Opt-in NPU regressions for independent requests. Uses only the Python stdlib."""
import argparse
import base64
import json
from pathlib import Path
import struct
import time
import unittest
import urllib.error
import urllib.request
import zlib

IMAGE_FORMAT = 'png'

def image_url(rgb):
    if IMAGE_FORMAT == 'jpeg':
        name = {(255, 0, 0): 'red', (0, 255, 0): 'green'}[rgb]
        data = (Path(__file__).parent / 'fixtures' / (name + '.jpg')).read_bytes()
        return 'data:image/jpeg;base64,' + base64.b64encode(data).decode()
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    pixels = (b'\x00' + bytes(rgb) * 32) * 32
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 32, 32, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


class SystemPromptCacheTests(unittest.TestCase):
    base_url = 'http://127.0.0.1:52625'
    model = 'gemma4-it:e2b'
    api_key = None
    records = []
    system = ('Return exactly the code from the CURRENT user message, nothing else. '
              'Reference data is not the user message: ' + 'Archived code OLD-001. ' * 200)

    def request(self, code, *, cache=True, think=False, system=None, stream=False, image=None, **extra):
        content = [{'type': 'text', 'text': f'Current code: {code}'}]
        if image is not None:
            content.append({'type': 'image_url', 'image_url': {'url': image}})
        body = {
            'model': self.model, 'cache_system_prompt': cache,
            'stream': stream, 'think': think, 'temperature': 0.0, 'top_k': 1,
            'max_tokens': 512 if think else 64, 'image-max-tokens': 560,
            'messages': [{'role': 'system', 'content': system or self.system},
                         {'role': 'user', 'content': content}], **extra,
        }
        if cache is None:
            body.pop('cache_system_prompt', None)
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        request = urllib.request.Request(self.base_url.rstrip('/') + '/v1/chat/completions',
                                         json.dumps(body).encode(), headers)
        started = time.monotonic()
        try:
            response = urllib.request.urlopen(request, timeout=180)
        except urllib.error.HTTPError as error:
            raise AssertionError(f'HTTP {error.code}: {json.load(error).get("error")}') from error
        with response:
            if stream:
                answer = ''
                usage = None
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    data = line[6:].strip()
                    if data == b'[DONE]':
                        break
                    event = json.loads(data)
                    if event.get('choices'):
                        answer += event['choices'][0].get('delta', {}).get('content') or ''
                    usage = event.get('usage') or usage
                result = {'choices': [{'message': {'content': answer}}], 'usage': usage}
            else:
                result = json.load(response)
        if 'error' in result:
            raise AssertionError(result['error'])
        self.records.append({'code': code, 'cache': cache, 'think': think,
                             'elapsed_seconds': time.monotonic() - started,
                             'stream': stream, 'usage': result.get('usage'),
                             'answer': result['choices'][0]['message']['content']})
        print(json.dumps(self.records[-1]), flush=True)
        return result

    @staticmethod
    def cached(result):
        return result['usage']['prompt_tokens_details']['cached_tokens']

    @staticmethod
    def answer(result):
        return result['choices'][0]['message']['content'].strip()

    def test_cold_warm_thinking_and_changing_media(self):
        for think in (False, True):
            with self.subTest(think=think):
                # Switching out of cached mode forces a fresh pin next time.
                self.request('NEW-101', cache=False, think=think)
                cold = self.request('NEW-101', think=think, image=image_url((255, 0, 0)))
                warm = self.request('NEW-202', think=think, image=image_url((0, 255, 0)))
                repeat = self.request('NEW-202', think=think, image=image_url((0, 255, 0)))
                uncached = self.request('NEW-202', cache=False, think=think, image=image_url((0, 255, 0)))
                self.assertEqual(self.cached(cold), 0)
                self.assertGreater(self.cached(warm), 500)
                self.assertEqual(self.cached(warm), self.cached(repeat))
                self.assertEqual(self.cached(uncached), 0)
                self.assertEqual(self.answer(cold), 'NEW-101')
                self.assertEqual(self.answer(warm), 'NEW-202')
                self.assertEqual(self.answer(warm), self.answer(uncached))
                self.assertEqual(self.answer(repeat), self.answer(uncached))
                # The larger image budget survived. Input includes at least
                # 400 new image/text tokens in addition to the cached prefix.
                self.assertGreater(warm['usage']['prompt_tokens'] - self.cached(warm), 400)

    def test_prefix_changes_streaming_and_default_mode(self):
        system = self.system + ' New reference version.'
        cold = self.request('NEW-303', system=system)
        warm = self.request('NEW-404', system=system, stream=True)
        self.assertEqual(self.cached(cold), 0)
        self.assertGreater(self.cached(warm), 500)
        self.assertEqual(self.answer(warm), 'NEW-404')
        # Explicit false and omission both use the existing conversation path.
        self.assertEqual(self.cached(self.request('NEW-505', cache=False)), 0)
        self.assertEqual(self.cached(self.request('NEW-606', cache=None)), 0)

    def test_changing_images_do_not_reuse_previous_pixels(self):
        system = 'Name the solid color in the image. Reply with one word: red or green.'
        self.request('ignored', cache=False, system=system)
        red = self.request('ignored', system=system, image=image_url((255, 0, 0)))
        green = self.request('ignored', system=system, image=image_url((0, 255, 0)))
        self.assertEqual(self.cached(red), 0)
        self.assertGreater(self.cached(green), 0)
        self.assertIn('red', self.answer(red).lower())
        self.assertIn('green', self.answer(green).lower())

    def test_invalid_history_and_option_do_not_poison_the_pin(self):
        self.request('NEW-701', cache=False)
        self.request('NEW-701')
        history = [{'role': 'system', 'content': self.system},
                   {'role': 'user', 'content': 'old observation'},
                   {'role': 'assistant', 'content': 'old answer'},
                   {'role': 'user', 'content': 'new observation'}]
        with self.assertRaisesRegex(AssertionError, 'HTTP 400: .*exactly one system'):
            self.request('ignored', messages=history)
        with self.assertRaisesRegex(AssertionError, 'HTTP 400: .*must be a boolean'):
            self.request('ignored', cache_system_prompt='true')
        warm = self.request('NEW-702')
        self.assertGreater(self.cached(warm), 500)
        self.assertEqual(self.answer(warm), 'NEW-702')

    def test_tool_schema_changes_invalidate_the_prefix(self):
        tools = [{'type': 'function', 'function': {
            'name': 'lookup_reference', 'description': 'Unused reference lookup.',
            'parameters': {'type': 'object', 'properties': {'key': {'type': 'string'}}},
        }}]
        cold = self.request('NEW-811', tools=tools, tool_choice='none')
        warm = self.request('NEW-812', tools=tools, tool_choice='none')
        self.assertEqual(self.cached(cold), 0)
        self.assertGreater(self.cached(warm), 500)
        self.assertEqual(self.answer(warm), 'NEW-812')
        tools[0]['function']['description'] = 'Updated reference lookup.'
        changed = self.request('NEW-813', tools=tools, tool_choice='none')
        self.assertEqual(self.cached(changed), 0)
        self.assertEqual(self.answer(changed), 'NEW-813')

    def test_oversized_observation_clears_the_pin_and_recovers(self):
        self.request('NEW-901')
        with self.assertRaisesRegex(AssertionError, 'HTTP 400: .*Max length'):
            self.request('oversized observation ' * 12000)
        cold = self.request('NEW-902')
        warm = self.request('NEW-903')
        self.assertEqual(self.cached(cold), 0)
        self.assertGreater(self.cached(warm), 500)
        self.assertEqual(self.answer(warm), 'NEW-903')

    def test_user_media_shape_changes_preserve_the_pin(self):
        self.request('NEW-921', cache=False)
        cold = self.request('NEW-921')
        image = self.request('NEW-922', image=image_url((255, 0, 0)))
        text = self.request('NEW-923')
        self.assertEqual(self.cached(cold), 0)
        self.assertGreater(self.cached(image), 500)
        self.assertEqual(self.cached(image), self.cached(text))
        self.assertEqual(self.answer(image), 'NEW-922')
        self.assertEqual(self.answer(text), 'NEW-923')

    def test_cancelled_generation_can_be_followed_by_a_fresh_observation(self):
        body = {
            'model': self.model, 'cache_system_prompt': True,
            'stream': True, 'think': True, 'temperature': 0.0,
            'max_tokens': 1024,
            'messages': [{'role': 'system', 'content': self.system},
                         {'role': 'user', 'content': 'Explain the reference data in detail, then list 100 archived codes.'}],
        }
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        request = urllib.request.Request(self.base_url.rstrip('/') + '/v1/chat/completions',
                                         json.dumps(body).encode(), headers)
        # Close while the response is still being generated. The server's
        # request queue serializes the subsequent observation behind this one.
        with urllib.request.urlopen(request, timeout=180) as response:
            for line in response:
                if line.startswith(b'data: ') and b'"delta"' in line:
                    break
        result = self.request('NEW-808', think=True)
        self.assertEqual(self.answer(result), 'NEW-808')
        warm = self.request('NEW-909', think=True)
        self.assertGreater(self.cached(warm), 500)
        self.assertEqual(self.answer(warm), 'NEW-909')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default=SystemPromptCacheTests.base_url)
    parser.add_argument('--model', default=SystemPromptCacheTests.model)
    parser.add_argument('--api-key')
    parser.add_argument('--image-format', choices=('png', 'jpeg'), default=IMAGE_FORMAT)
    args, remaining = parser.parse_known_args()
    SystemPromptCacheTests.base_url = args.base_url
    SystemPromptCacheTests.model = args.model
    SystemPromptCacheTests.api_key = args.api_key
    IMAGE_FORMAT = args.image_format
    unittest.main(argv=['test_server.py', *remaining])
