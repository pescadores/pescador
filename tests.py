import pescador


def test_buffer_stream():
    stream = pescador.buffer_stream(range(3), 2, 1)
    for exp, act in zip([[0, 1]], stream):
        assert exp == act

    stream = pescador.buffer_stream(range(3), 2, 2)
    for exp, act in zip([[0, 1], [2]], stream):
        assert exp == act
