import redis
import os

class RedisClient:
    def __init__(self):
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", 6379))
        self.client = redis.Redis(host=self.host, port=self.port, decode_responses=True)

    def get_redirect(self, term):
        val = self.client.get(f"r:{term}")
        return val if val else term

    def get_entities(self, anchor):
        return self.client.lrange(f"a:{anchor}", 0, -1)

    def get_type(self, entity):
        # Key: "t:EntityName"
        val = self.client.get(f"t:{entity}")
        if val:
            return [val]  # Return as list e.g. ['SoccerClub']
        return ["Thing"]  # Default fallback
