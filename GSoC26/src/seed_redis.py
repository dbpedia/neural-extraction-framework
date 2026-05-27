import redis
import os

def seed_data():
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379)) # Uses environment variable now
    
    r = redis.Redis(host=host, port=port, decode_responses=True)

    print("⚠️ Warning: Flushing database to ensure clean state...")
    r.flushdb() 
    
    print("Seeding Redis with GSoC25-Style Data...")
    
    # 1. REDIRECTS (r:Slang -> Formal)
    r.set("r:Barca", "FC Barcelona")
    r.set("r:Man City", "Manchester City F.C.")
    r.set("r:CR7", "Cristiano Ronaldo")
    
    # 2. ANCHOR CANDIDATES (a:Text -> List[Entities])
    r.rpush("a:Al-Nassr", "Al-Nassr FC", "Al-Nasr SC")
    r.rpush("a:Cristiano Ronaldo", "Cristiano Ronaldo")
    r.rpush("a:La Liga", "La Liga")
    r.rpush("a:Premier League", "Premier League")
    r.rpush("a:Man City", "Manchester City F.C.")
    r.rpush("a:2015", "2015") # Ensure 2015 exists as an entity
    r.rpush("a:Ballon d'Or", "Ballon d'Or")
    r.rpush("a:Lionel Messi", "Lionel Messi")
    r.rpush("a:Chicago Bulls", "Chicago Bulls")
    r.rpush("a:Real Madrid", "Real Madrid CF")

    # 3. TYPES (t:Entity -> Type)
    r.set("t:Cristiano Ronaldo", "Athlete")
    r.set("t:Lionel Messi", "Athlete")
    r.set("t:Al-Nasr SC", "SoccerClub")
    r.set("t:Al-Nassr FC", "SoccerClub")       
    r.set("t:FC Barcelona", "SoccerClub")
    r.set("t:Real Madrid CF", "SoccerClub")
    r.set("t:La Liga", "SoccerLeague")
    r.set("t:Manchester City F.C.", "SoccerClub")
    r.set("t:Premier League", "SoccerLeague")
    r.set("t:2015", "Year") 
    r.set("t:Ballon d'Or", "Award")
    r.set("t:Chicago Bulls", "BasketballTeam")
    
    print("Redis Seeded! (Clean slate guaranteed)")

if __name__ == "__main__":
    seed_data()
