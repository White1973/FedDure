# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: easyfl/pb/client_service.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from easyfl.pb import common_pb2 as easyfl_dot_pb_dot_common__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1e\x65\x61syfl/pb/client_service.proto\x12\teasyfl.pb\x1a\x16\x65\x61syfl/pb/common.proto\"\x85\x01\n\x0eOperateRequest\x12&\n\x04type\x18\x01 \x01(\x0e\x32\x18.easyfl.pb.OperationType\x12\r\n\x05model\x18\x02 \x01(\x0c\x12\x12\n\ndata_index\x18\x03 \x01(\x05\x12(\n\x06\x63onfig\x18\x04 \x01(\x0b\x32\x18.easyfl.pb.OperateConfig\"\xce\x01\n\rOperateConfig\x12\x12\n\nbatch_size\x18\x01 \x01(\x05\x12\x13\n\x0blocal_epoch\x18\x02 \x01(\x05\x12\x0c\n\x04seed\x18\x03 \x01(\x03\x12\'\n\toptimizer\x18\x04 \x01(\x0b\x32\x14.easyfl.pb.Optimizer\x12\x12\n\nlocal_test\x18\x05 \x01(\x08\x12\x0f\n\x07task_id\x18\x06 \x01(\t\x12\x10\n\x08round_id\x18\x07 \x01(\x05\x12\r\n\x05track\x18\x08 \x01(\x08\x12\x17\n\x0ftest_batch_size\x18\t \x01(\x05\"7\n\tOptimizer\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\n\n\x02lr\x18\x02 \x01(\x02\x12\x10\n\x08momentum\x18\x03 \x01(\x02\"4\n\x0fOperateResponse\x12!\n\x06status\x18\x01 \x01(\x0b\x32\x11.easyfl.pb.Status*4\n\rOperationType\x12\x11\n\rOP_TYPE_TRAIN\x10\x00\x12\x10\n\x0cOP_TYPE_TEST\x10\x01\x32S\n\rClientService\x12\x42\n\x07Operate\x12\x19.easyfl.pb.OperateRequest\x1a\x1a.easyfl.pb.OperateResponse\"\x00\x62\x06proto3')

_OPERATIONTYPE = DESCRIPTOR.enum_types_by_name['OperationType']
OperationType = enum_type_wrapper.EnumTypeWrapper(_OPERATIONTYPE)
OP_TYPE_TRAIN = 0
OP_TYPE_TEST = 1


_OPERATEREQUEST = DESCRIPTOR.message_types_by_name['OperateRequest']
_OPERATECONFIG = DESCRIPTOR.message_types_by_name['OperateConfig']
_OPTIMIZER = DESCRIPTOR.message_types_by_name['Optimizer']
_OPERATERESPONSE = DESCRIPTOR.message_types_by_name['OperateResponse']
OperateRequest = _reflection.GeneratedProtocolMessageType('OperateRequest', (_message.Message,), {
  'DESCRIPTOR' : _OPERATEREQUEST,
  '__module__' : 'easyfl.pb.client_service_pb2'
  # @@protoc_insertion_point(class_scope:easyfl.pb.OperateRequest)
  })
_sym_db.RegisterMessage(OperateRequest)

OperateConfig = _reflection.GeneratedProtocolMessageType('OperateConfig', (_message.Message,), {
  'DESCRIPTOR' : _OPERATECONFIG,
  '__module__' : 'easyfl.pb.client_service_pb2'
  # @@protoc_insertion_point(class_scope:easyfl.pb.OperateConfig)
  })
_sym_db.RegisterMessage(OperateConfig)

Optimizer = _reflection.GeneratedProtocolMessageType('Optimizer', (_message.Message,), {
  'DESCRIPTOR' : _OPTIMIZER,
  '__module__' : 'easyfl.pb.client_service_pb2'
  # @@protoc_insertion_point(class_scope:easyfl.pb.Optimizer)
  })
_sym_db.RegisterMessage(Optimizer)

OperateResponse = _reflection.GeneratedProtocolMessageType('OperateResponse', (_message.Message,), {
  'DESCRIPTOR' : _OPERATERESPONSE,
  '__module__' : 'easyfl.pb.client_service_pb2'
  # @@protoc_insertion_point(class_scope:easyfl.pb.OperateResponse)
  })
_sym_db.RegisterMessage(OperateResponse)

_CLIENTSERVICE = DESCRIPTOR.services_by_name['ClientService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _OPERATIONTYPE._serialized_start=525
  _OPERATIONTYPE._serialized_end=577
  _OPERATEREQUEST._serialized_start=70
  _OPERATEREQUEST._serialized_end=203
  _OPERATECONFIG._serialized_start=206
  _OPERATECONFIG._serialized_end=412
  _OPTIMIZER._serialized_start=414
  _OPTIMIZER._serialized_end=469
  _OPERATERESPONSE._serialized_start=471
  _OPERATERESPONSE._serialized_end=523
  _CLIENTSERVICE._serialized_start=579
  _CLIENTSERVICE._serialized_end=662
# @@protoc_insertion_point(module_scope)
