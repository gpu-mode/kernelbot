from app import add_medals

def test_add_medals():
    # Test with user objects including scores
    two_users = [
        {"user_name": "alice", "score": 0.123},
        {"user_name": "bob", "score": 0.124}
    ]
    assert add_medals(two_users) == [
        ("🥇alice", "123000.000μs"),
        ("🥈bob", "124000.000μs"),
        ("", "")
    ]
    
    # Test with more than 3 users
    four_users = [
        {"user_name": "alice", "score": 0.123},
        {"user_name": "bob", "score": 0.124},
        {"user_name": "carol", "score": 0.125},
        {"user_name": "dave", "score": 0.126}
    ]
    assert add_medals(four_users) == [
        ("🥇alice", "123000.000μs"),
        ("🥈bob", "124000.000μs"),
        ("🥉carol", "125000.000μs")
    ]