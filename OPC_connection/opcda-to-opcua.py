# Name: opcda_to_opcua.py
# Original Author: Eskild Schroll-Fleischer <esksch@dtu.dk>
# Modified by: Khush Harman
# Date: 30th of August 2017
#
# Description:
# Proxy between OPC-DA server and OPC-UA client.
# Firstly the OPC-DA namespace is traversed using a recursive
# function. These variables are then classified as readable or writable
# and added to the OPC-UA namespace. The readable variables are read
# periodically from the OPC-DA server and published on the OPC-UA server.
# The writable OPC-UA tags are monitored for changes. When a change is
# caught then the new value is published to the OPC-DA server.
#
# The code is organized as follows:
# 1. Configuration
# 2. Connect to OPC-DA server
# 3. Discover OPC-DA server nodes
# 4. Subscribe to datachanges coming from OPC-UA clients
# 5. Read all readables simultaneously and update the OPC-UA variables
#    to reflect the OPC-DA readings

# Requires Anaconda, OpenOPC
# L 609 in address_space.py, python-opcua v 0.90.3

import decimal
import logging
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import OpenOPC
import pandas as pd
import pywintypes
import tqdm
from opcua import Server, ua, uamethod, Client
from opcua.common.xmlexporter import XmlExporter

# 1. Configuration
OPC_DA_SERVER = 'RSLinx Remote OPC Server'
# OPC_UA_CERTIFICATE = 'certificate.der'
# OPC_UA_PRIVATE_KEY = 'private_key.pem'
OPC_UA_URI = 'http://examples.freeopcua.github.io'

pywintypes.datetime = pywintypes.TimeType

# Constants
ITEM_ID_VIRTUAL_PROPERTY = 0
ITEM_CANONICAL_DATATYPE = 1
ITEM_VALUE = 2
ITEM_QUALITY = 3
ITEM_TIMESTAMP = 4
ITEM_ACCESS_RIGHTS = 5
SERVER_SCAN_RATE = 6
ITEM_EU_TYPE = 7
ITEM_EU_INFO = 8
ITEM_DESCRIPTION = 101
ACCESS_READ = 0
ACCESS_WRITE = 1
ACCESS_READ_WRITE = 2
VARIABLE_HANDLES = 'variable_handles.csv'
SERVER_CONFIG = 'server.xml'

# Set up server
server = Server()
server.set_endpoint('opc.tcp://localhost:4843/freeopcua/server')
# server.load_certificate(OPC_UA_CERTIFICATE)
# server.load_private_key(OPC_UA_PRIVATE_KEY)
uri = OPC_UA_URI
dir_path = os.path.abspath(os.path.dirname(__file__))


# define output_argument for custom method
out_arg = ua.Argument()
out_arg.Name = "status"
out_arg.DataType = ua.NodeId(ua.ObjectIds.Int64)
out_arg.ValueRank = -1
out_arg.ArrayDimensions = []
status_code = ""


@uamethod
def get_status(parent):
    return status_code


# 2. Connect to OPC-DA server
c = OpenOPC.client()
# List OPC-DA servers
servers = c.servers()

c.connect(OPC_DA_SERVER)


class SubscriptionHandler(object):
    def __init__(self, n):
        self.i = 0
        self.n = n

    def final_datachange_notification(self, node, val, data):
        path_as_string = node.get_path(as_string=True)
        # 'path_as_string' is a list of strings containing:
        # 0: 0:Root
        # 1: 1:Objects
        # 2: 2:OPC DA Server
        # 3 and onwards: 3:[Step of path to node in OPC-DA]
        opc_da_address = '.'.join([a.split(':', 1)[1]
                                    for a in path_as_string[3:]])
        cc = OpenOPC.client()
        cc.connect(OPC_DA_SERVER)
        global status_code
        status_code = cc.write((opc_da_address, val,))

        print('Datachange', opc_da_address, val, status_code)
        cc.close()
    # This function is called initially to catch the notifications from newly added nodes

    def datachange_notification(self, node, val, data):
        self.i = self.i + 1
        # print('Catching meaningless datachange notification')
        if self.i == self.n:
            # print('Finished catching meaningless datachange notifications')
            self.datachange_notification = self.final_datachange_notification


def read_value(value):
    value = value[0]
    if isinstance(value, decimal.Decimal):
        value = float(value)
    elif isinstance(value, list):
        if len(value) == 0:
            value = None
    elif isinstance(value, tuple):
        if len(value) == 0:
            value = None
    return value


def load_variable_handles(idx, root):

    # 'nodes' is a list of dot-delimited strings.
    nodes = c.list('*', recursive=True)
    readable_variable_handles = {}
    writeable_variable_handles = {}
    values = {}

    tree = {}

    for node in tqdm.tqdm(nodes):

        parts = node.split('.')
        # Folders are the steps on the path to the file.
        folders = parts[:-1]
        file = parts[-1]
        # Create folder tree if it does not already exist
        for i, folder in enumerate(folders, 1):
            if i == 1:
                parent = root
            else:
                parent = tree[path]
            path = '.'.join(folders[0:i])
            if path not in tree.keys():
                tree[path] = parent.add_folder(idx, folder)

        # 'path' is now the folder that file resides in.
        # Determine node properties
        node_obj = {}
        value = c.properties(node, id=5)

        if value == 'Read':
            value = ACCESS_READ
        elif value == 'Write':
            value = ACCESS_WRITE
        elif value == 'Read/Write':
            value = ACCESS_READ_WRITE

        node_obj[ITEM_ACCESS_RIGHTS] = value

        node_obj[ITEM_VALUE], node_obj[ITEM_QUALITY], node_obj[ITEM_TIMESTAMP] = c.read(
            node)

        node_obj[ITEM_CANONICAL_DATATYPE] = c.properties(node, id=1)

        current_value = read_value((node_obj[ITEM_VALUE],))
        if type(current_value) == int:
            variant_type = ua.VariantType.UInt32
        elif type(current_value) == float:
            variant_type = ua.VariantType.Float
        else:
            current_value = 0
            variant_type = ua.VariantType.UInt32
            
        opcua_node = tree[path].add_variable(
            idx, file, ua.Variant(current_value, variant_type))
        # Determine readable vs. writable

        if node_obj[ITEM_ACCESS_RIGHTS] in [ACCESS_READ]:
            readable_variable_handles[node] = opcua_node

        if node_obj[ITEM_ACCESS_RIGHTS] in [ACCESS_WRITE, ACCESS_READ_WRITE]:
            opcua_node.set_writable()
            writeable_variable_handles[node] = opcua_node

        values[node] = value
        # print('Adding node '+str(file)+' at path '+path)
    return readable_variable_handles, writeable_variable_handles, values


def main():
    try:
        # 3. Discover OPC-DA server nodes
        if os.getcwd() != dir_path:

            global VARIABLE_HANDLES, SERVER_CONFIG
            VARIABLE_HANDLES = os.path.join(dir_path, VARIABLE_HANDLES)
            SERVER_CONFIG = os.path.join(dir_path, SERVER_CONFIG)

            """Saving and loading data from xml/csv file for tags (Unresolved)"""

            # # Import server nodes from xml, otherwise server doesn't recognise readable and writable tags.
            # idx = server.register_namespace(uri)
            # server.import_xml(SERVER_CONFIG)

            # # load readable/writables from csv
            # df = pd.read_csv(VARIABLE_HANDLES)

            # # split readables and writeables
            # df_readable = df[df.iloc[:, 2] == ACCESS_READ]
            # df_writable = pd.concat(
            #     [df[df.iloc[:, 2] == ACCESS_READ_WRITE], (df[(df.iloc[:, 2] == ACCESS_WRITE)])])

            # df_readable = df_readable.reset_index(
            #     drop=True).set_index('Tag Name')
            # df_writable = df_writable.reset_index(
            #     drop=True).set_index('Tag Name')

            # # Map readable/writable variable handle tag names to node objects
            # df_readable = df_readable['Node ID'].map(
            #     lambda x: server.get_node(x))
            # df_writable = df_writable['Node ID'].map(
            #     lambda x: server.get_node(x))

       
            # readable_variable_handles = dict(df_readable)
            # writeable_variable_handles = dict(df_writable)

        idx = server.register_namespace(uri)
        
        
        root = server.nodes.objects.add_object(idx, OPC_DA_SERVER)
        root.add_method(idx, "get_status", get_status, [], [out_arg])

        readable_variable_handles, writeable_variable_handles, values = load_variable_handles(
            idx, root)
        
        
        server.export_xml_by_ns(SERVER_CONFIG, server.get_namespace_array())

        d = {**readable_variable_handles, **writeable_variable_handles}
        df = pd.DataFrame({'Tag Name': d.keys(), 'Node ID': d.values(),
                            'Access Rights': values.values()}).set_index('Tag Name')
        df.to_csv(VARIABLE_HANDLES)

        server.start()

        # 4. Subscribe to datachanges coming from OPC-UA clients
        handler = SubscriptionHandler(len(writeable_variable_handles))
        sub = server.create_subscription(100, handler).subscribe_data_change(
            writeable_variable_handles.values())
        readables = list(readable_variable_handles.keys())

        while True:

            time.sleep(0.1)
            # 5. Read all readables simultaneously and update the OPC-UA variables

            for reading in c.read(readables, timeout=36000, update=100):
                opc_da_id = reading[0]

                variable_handle = readable_variable_handles[opc_da_id]
                variable_handle.set_value(read_value(reading[1:]))

    except Exception as e:
        print(e)

    finally:
        server.stop()
        c.close()


if __name__ == "__main__":
    main()
