from app import add_medals

def test_add_medals():
    # Test with user objects including scores
    two_users = [
        {"user_name": "alice", "score": 0.123},
        {"user_name": "bob", "score": 0.124}
    ]
    assert add_medals(two_users) == [
        ("🥇alice", "1.23e+05μs"),
        ("🥈bob", "1.24e+05μs"),
        ("", "")
    ]
    
    # Test with more than 3 users
    three_users = [
        {"user_name": "alice", "score": 0.123},
        {"user_name": "bob", "score": 0.124},
        {"user_name": "carol", "score": 0.125},
        {"user_name": "dave", "score": 0.126}
    ]
    assert add_medals(three_users) == [
        ("🥇alice", "1.23e+05μs"),
        ("🥈bob", "1.24e+05μs"),
        ("🥉carol", "1.25e+05μs")
    ]