class Point:
    def __init__(self, lat, long):
        self.lat = lat
        self.long = long


def locate(latitude: float, longitude: float) -> Point:
    p = Point(latitude, longitude)
    return p


def chang():
    """
    docstring 확인하기
    """
    print(chang)


if __name__ == "__main__":
    """
    annotation 활용하기 , docstring 활용하기
    """
    print(chang.__doc__)
    print(locate.__annotations__)
    print(locate(3.45, 4.56))