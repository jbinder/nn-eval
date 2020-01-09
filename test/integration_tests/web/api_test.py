import unittest

from web.api import app


# noinspection PyMethodMayBeStatic
class ApiTest(unittest.TestCase):
    def test_predict(self):
        app.config['TESTING'] = True
        client = app.test_client()
        rv = client.post('/', json={'input': [2]})
        response = rv.get_json()
        self.assertAlmostEquals(response['result'][0], 6, 4)
