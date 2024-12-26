from typing import Dict, Union, Optional
from timeit import default_timer as timer

class SingleTimer():
    def __init__(self):
        self.start_time: Optional[float] = timer()
        self.stop_time: Optional[float] = None

    @property
    def current_time(self) -> float:
        if self.start_time is None:
            raise ValueError("Timer is not started")

        stop = timer()
        return (stop - self.start_time) * 1e3

    def start(self) -> None:
        self.start_time = timer()

    def get_time_and_restart(self) -> float:
        t = self.current_time
        self.start_time = timer()
        return t

    def stop(self) -> None:
        self.stop_time = timer()

    def get_start_to_stop(self) -> float:
        return (self.stop_time - self.start_time) * 1e3
    
    def __repr__(self):
        return str(self.current_time)


class MultipleTimer():
    """
    Collection of timers Unit in ms
    """
    def __init__(self, each_timer_name: Optional[str] = None):
        self.__INIT_TIME = timer()
        self.__timers_collection: Dict[str, SingleTimer] = {}
        self.stop_time: Optional[float] = None
        self.newTimer("main")

        if each_timer_name is not None:
            for timer_name in each_timer_name:
                self.newTimer(timer_name)

        self.restart_all()

    @property
    def timer(self) -> SingleTimer:
        return self.__timers_collection

    @property
    def main(self) -> SingleTimer:
        return self.__timers_collection["main"]

    def get_all_time(self) -> Dict[str, float]:
        stop_time = timer()
        return {name: (stop_time - self.__timers_collection[name].start_time)*1e3 for name in self.__timers_collection.keys()}

    def newTimer(self, name: str) -> None:
        if name not in self.__timers_collection.keys():
            self.__timers_collection[name] = SingleTimer()

    def stop_all(self) -> None:
        self.stop_time = timer()
        for each_timer in self.__timers_collection.keys():
            self.__timers_collection[each_timer].stop_time = self.stop_time

    def restart_all(self) -> None:
        start = timer()
        for each_timer in self.__timers_collection.keys():
            self.__timers_collection[each_timer].start_time = start

    def __repr__(self) -> None:
        return str(self.get_all_time())

