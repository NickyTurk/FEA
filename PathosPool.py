from pathos.helpers import mp
from pathos.multiprocessing import ProcessPool


class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class NoDaemonProcessPool(ProcessPool):
    Process = NoDaemonProcess