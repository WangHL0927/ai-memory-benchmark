"""实验子进程只允许回环网络；阻止遥测、自动更新和本地模型下载。"""
import ipaddress
import socket


def _host(host):
    if host not in (None, '', b'', 'localhost', b'localhost'):
        try:
            allowed = ipaddress.ip_address(host).is_loopback
        except ValueError:
            allowed = False
        if not allowed:
            raise OSError('EVAL_NETWORK_DENIED')


def _local(address):
    if isinstance(address, tuple):
        _host(address[0])


_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
_getaddrinfo = socket.getaddrinfo
_sendto = socket.socket.sendto


def connect(self, address):
    _local(address)
    return _connect(self, address)


def connect_ex(self, address):
    _local(address)
    return _connect_ex(self, address)


def getaddrinfo(host, *args, **kwargs):
    _host(host)
    return _getaddrinfo(host, *args, **kwargs)


def sendto(self, data, *args):
    _local(args[-1])
    return _sendto(self, data, *args)


socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
socket.getaddrinfo = getaddrinfo
socket.socket.sendto = sendto
