from opcua import ua, Client
from OpenOPC import type_check
import time

OPC_DA_SERVER = 'RSLinx Remote OPC Server'


class OPCWrapper:

    def __init__(self, client, uri):
        self.client = client
        self.uri = uri
        self.idx = client.get_namespace_index(uri)
        self.root = client.get_root_node()
        self.obj = self._get_server_obj()

    def read(self, tags, *args, **kwargs):
        """
        Basic implementation of opc read to work as a wrapper method for OpenOPC using opcua.

        """
        tag_list, single, valid = type_check(tags)
        if not valid:
            raise TypeError("Invalid tags entered, try again")

        nodes_list = []
        for tag in tag_list:
            # perform opcua read
            node = self._get_node(tag)

            if node:
                # return list of tuples (name, value, quality, timestamp) if not single else (value, quality, timestamp)
                # since we are only using value and timestamp, for now only these values are returned, rest are None

                data_value = node.get_data_value()

                nodes_list.append((node.get_browse_name().Name,
                                   data_value.Value.Value,
                                   data_value.StatusCode.name,
                                   str(data_value.SourceTimestamp)))
                if single:
                    return nodes_list[0][1:]
            else:
                raise TypeError(f"Invalid tag path: {tag} , please try again")

        return nodes_list

    def write(self, tag_value_pairs, *args, **kwargs):
        """
        Basic implementation of opc write compatible with opcua
        
        """

        if type(tag_value_pairs) in (list, tuple) and type(tag_value_pairs[0]) in (list, tuple):
            single = False
        else:
            single = True
            tag_value_pairs = [tag_value_pairs]

        for tag, value in tag_value_pairs:
            
            node = self._get_node(tag)
            if node:
                # return status code(s) for all (tag, value) pairs
                try:
                    node.set_value(value)

                    while not self._get_status():
                        # try again
                        print("Tag write unsuccessful, trying again...")
                        time.sleep(0.5)
                        node.set_value(value)
                    
                except Exception as e:
                    print(e)

    def _get_server_obj(self):
        return self.root.get_child([f"{str(self.root.get_browse_name().NamespaceIndex)}:Objects",
                       f"{str(self.idx)}:{OPC_DA_SERVER}"])

    def _get_status(self):
        return self.obj.call_method(f"{str(self.idx)}:get_status") 

    def _get_node(self, path):
        """
        Returns node object as a string for the node specified by given path
        path format: Objects_Folder.Server_Folder.Nodes_Folder.Node

        """
        if isinstance(path, str) and path:
            # add object folder to path and convert path to correct format
            # correct path format example: "0:Objects","2:RSLinx Remote OPC Server", "2:[FluidMech]PumpDrivePwrFlx525", "2:OCmd_Start"
            # check if path returns NodeId i.e. correct path

            folders = [f"{str(self.root.get_browse_name().NamespaceIndex)}:Objects",
                       f"{str(self.idx)}:{OPC_DA_SERVER}"]

            for folder in path.split("."):
                folder = f"{str(self.idx)}:{folder}"
                folders.append(folder)

            return self.root.get_child(folders)

    def close(self):
        self.client.disconnect()
