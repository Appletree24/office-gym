"""Local Unix socket world service. Episode/principal are fixed by server startup.

This internal protocol is not MCP. OS-level worker filesystem isolation must be
configured by the experiment launcher independently of this transport.
"""
import json
import os
import socket
from pathlib import Path
from officegym.env.world import World, canonical
from officegym.tools.dispatcher import Dispatcher
from officegym.tools.schema import tool_schemas

MAX_REQUEST = 1024 * 1024


def serve(world_path, socket_path, max_calls=None):
    socket_path = Path(socket_path)
    if socket_path.exists():
        raise FileExistsError(socket_path)
    world = World(world_path)
    listener = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    try:
        listener.bind(str(socket_path))
        os.chmod(socket_path,0o600)
        listener.listen(8)
        dispatcher = Dispatcher(world,max_calls=max_calls)
        while True:
            connection,_ = listener.accept()
            with connection:
                connection.settimeout(30)
                stream = connection.makefile('rb')
                try:
                    encoded = stream.readline(MAX_REQUEST+1)
                    if len(encoded)>MAX_REQUEST or not encoded.endswith(b'\n'):
                        raise ValueError('invalid request frame')
                    request = json.loads(encoded)
                    if not isinstance(request,dict):
                        raise ValueError('invalid request')
                    if request == {'method':'tools/list'}:
                        response = {'tools':tool_schemas()}
                    elif request.get('method') == 'tools/call' and set(request)<= {'method','name','arguments','operation_id'}:
                        response = dispatcher.call(request.get('name'),request.get('arguments'),request.get('operation_id'))
                    else:
                        raise ValueError('invalid method or trusted context injection')
                except (ValueError,TypeError,socket.timeout):
                    response = {'protocol_error':'INVALID_REQUEST'}
                connection.sendall((canonical(response)+'\n').encode())
    finally:
        listener.close()
        world.close()
        if socket_path.exists():
            socket_path.unlink()


class Client:
    def __init__(self,socket_path,timeout=30):
        self.socket_path,self.timeout = str(socket_path),timeout

    def request(self,payload):
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(self.socket_path)
            connection.sendall((canonical(payload)+'\n').encode())
            with connection.makefile('rb') as stream:
                line = stream.readline(16*MAX_REQUEST+1)
            if not line.endswith(b'\n') or len(line)>16*MAX_REQUEST:
                raise RuntimeError('invalid world response frame')
            return json.loads(line)

    def call(self,name,arguments,operation_id=None):
        return self.request({'method':'tools/call','name':name,'arguments':arguments,'operation_id':operation_id})
