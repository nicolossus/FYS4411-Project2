from benchmark_tools import timer


@timer
def waste_some_time(num_times):
    for _ in range(num_times):
        sum([i**2 for i in range(10000)])


if __name__ == "__main__":
    waste_some_time(100)
