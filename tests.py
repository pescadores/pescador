import pescador


def test_stream_buffer():
    stream = pescador.stream_buffer(range(3), 2, 1)
    for exp, act in zip([[0, 1]], stream):
        assert exp == act

    stream = pescador.stream_buffer(range(3), 2, 2)
    for exp, act in zip([[0, 1], [2]], stream):
        assert exp == act
