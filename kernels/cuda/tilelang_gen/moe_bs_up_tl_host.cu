// tilelang target: {"kind":"c","tag":"","keys":["cpu"]}
#define TVM_EXPORTS
#include "tvm/runtime/base.h"
#include "tvm/runtime/c_backend_api.h"
#include "tvm/ffi/c_api.h"
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#if defined(_MSC_VER)
#define TL_ALIGN(N) __declspec(align(N))
#else
#define TL_ALIGN(N) __attribute__((aligned(N)))
#endif
#ifdef __OBJC__
#include "tvm/runtime/device_api.h"
#include "tvm/ffi/function.h"
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <torch/mps.h>
#endif
void* __tvm_ffi__library_ctx = NULL;
static void* __tvm_set_device_packed = NULL;
static void* __tvm_tensormap_create_tiled_packed = NULL;
static void* main_kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self_handle, void* args, int32_t num_args, void* result);
#ifdef __cplusplus
extern "C"
#endif
int32_t TVMFFIEnvTensorAlloc(void*, void*);
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self_handle, void* args, int32_t num_args, void* result) {
  TL_ALIGN(128) TVMFFIAny stack[21];
  void* stack_ffi_any = stack;
  if (!((num_args == 8))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "main: num_args should be 8", (long long)(num_args), (long long)(8));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main: args pointer is NULL");
    return -1;
  }
  int32_t A_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((A_handle_type_index == 0) || (A_handle_type_index == 4)) || (A_handle_type_index == 7)) || (64 <= A_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input A expected pointer or tensor handle");
    return -1;
  }
  int32_t W1_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((W1_handle_type_index == 0) || (W1_handle_type_index == 4)) || (W1_handle_type_index == 7)) || (64 <= W1_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input W1 expected pointer or tensor handle");
    return -1;
  }
  int32_t W3_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((W3_handle_type_index == 0) || (W3_handle_type_index == 4)) || (W3_handle_type_index == 7)) || (64 <= W3_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input W3 expected pointer or tensor handle");
    return -1;
  }
  int32_t SFA_handle_type_index = (((TVMFFIAny*)args)[3].type_index);
  if (!(((((SFA_handle_type_index == 0) || (SFA_handle_type_index == 4)) || (SFA_handle_type_index == 7)) || (64 <= SFA_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input SFA expected pointer or tensor handle");
    return -1;
  }
  int32_t SFW1_handle_type_index = (((TVMFFIAny*)args)[4].type_index);
  if (!(((((SFW1_handle_type_index == 0) || (SFW1_handle_type_index == 4)) || (SFW1_handle_type_index == 7)) || (64 <= SFW1_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input SFW1 expected pointer or tensor handle");
    return -1;
  }
  int32_t SFW3_handle_type_index = (((TVMFFIAny*)args)[5].type_index);
  if (!(((((SFW3_handle_type_index == 0) || (SFW3_handle_type_index == 4)) || (SFW3_handle_type_index == 7)) || (64 <= SFW3_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input SFW3 expected pointer or tensor handle");
    return -1;
  }
  int32_t Eid_handle_type_index = (((TVMFFIAny*)args)[6].type_index);
  if (!(((((Eid_handle_type_index == 0) || (Eid_handle_type_index == 4)) || (Eid_handle_type_index == 7)) || (64 <= Eid_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Eid expected pointer or tensor handle");
    return -1;
  }
  int32_t allocator_anchor_type_index = (((TVMFFIAny*)args)[7].type_index);
  if (!((allocator_anchor_type_index == 70))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "main: allocator anchor must be a Tensor", (long long)(allocator_anchor_type_index), (long long)(70));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!((((TVMFFIAny*)args)[7].v_ptr) == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main: allocator anchor is NULL");
    return -1;
  }
  if (!(((((DLTensor*)((void*)((char*)(((TVMFFIAny*)args)[7].v_ptr) + (int64_t)24)))[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "main: allocator anchor has the wrong device type", (long long)((((DLTensor*)((void*)((char*)(((TVMFFIAny*)args)[7].v_ptr) + (int64_t)24)))[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t dev_id = (((DLTensor*)((void*)((char*)(((TVMFFIAny*)args)[7].v_ptr) + (int64_t)24)))[0].device.device_id);
  void* A_handle = ((A_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* W1_handle = ((W1_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* W3_handle = ((W3_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  void* SFA_handle = ((SFA_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[3].v_ptr) + 24)) : (((TVMFFIAny*)args)[3].v_ptr));
  void* SFW1_handle = ((SFW1_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[4].v_ptr) + 24)) : (((TVMFFIAny*)args)[4].v_ptr));
  void* SFW3_handle = ((SFW3_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[5].v_ptr) + 24)) : (((TVMFFIAny*)args)[5].v_ptr));
  void* Eid_handle = ((Eid_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[6].v_ptr) + 24)) : (((TVMFFIAny*)args)[6].v_ptr));
  bool main_A_is_null = (A_handle == NULL);
  if (!(!main_A_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.A is expected to have non-NULL pointer");
    return -1;
  }
  bool main_W1_is_null = (W1_handle == NULL);
  if (!(!main_W1_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.W1 is expected to have non-NULL pointer");
    return -1;
  }
  bool main_W3_is_null = (W3_handle == NULL);
  if (!(!main_W3_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.W3 is expected to have non-NULL pointer");
    return -1;
  }
  bool main_SFA_is_null = (SFA_handle == NULL);
  if (!(!main_SFA_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.SFA is expected to have non-NULL pointer");
    return -1;
  }
  bool main_SFW1_is_null = (SFW1_handle == NULL);
  if (!(!main_SFW1_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.SFW1 is expected to have non-NULL pointer");
    return -1;
  }
  bool main_SFW3_is_null = (SFW3_handle == NULL);
  if (!(!main_SFW3_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.SFW3 is expected to have non-NULL pointer");
    return -1;
  }
  bool main_Eid_is_null = (Eid_handle == NULL);
  if (!(!main_Eid_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.Eid is expected to have non-NULL pointer");
    return -1;
  }
  void* main_A_shape = (((DLTensor*)A_handle)[0].shape);
  void* main_W1_shape = (((DLTensor*)W1_handle)[0].shape);
  void* main_W3_shape = (((DLTensor*)W3_handle)[0].shape);
  void* main_SFA_shape = (((DLTensor*)SFA_handle)[0].shape);
  void* main_SFW1_shape = (((DLTensor*)SFW1_handle)[0].shape);
  void* main_SFW3_shape = (((DLTensor*)SFW3_handle)[0].shape);
  void* main_Eid_shape = (((DLTensor*)Eid_handle)[0].shape);
  if (!(((((DLTensor*)A_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A ndim mismatch, expected 2", (long long)((((DLTensor*)A_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_A_strides = (((DLTensor*)A_handle)[0].strides);
  void* A = (((DLTensor*)A_handle)[0].data);
  if (!(((((DLTensor*)W1_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W1 ndim mismatch, expected 3", (long long)((((DLTensor*)W1_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_W1_strides = (((DLTensor*)W1_handle)[0].strides);
  void* W1 = (((DLTensor*)W1_handle)[0].data);
  if (!(((((DLTensor*)W3_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W3 ndim mismatch, expected 3", (long long)((((DLTensor*)W3_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_W3_strides = (((DLTensor*)W3_handle)[0].strides);
  void* W3 = (((DLTensor*)W3_handle)[0].data);
  if (!(((((DLTensor*)SFA_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFA ndim mismatch, expected 1", (long long)((((DLTensor*)SFA_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_SFA_strides = (((DLTensor*)SFA_handle)[0].strides);
  void* SFA = (((DLTensor*)SFA_handle)[0].data);
  if (!(((((DLTensor*)SFW1_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFW1 ndim mismatch, expected 2", (long long)((((DLTensor*)SFW1_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_SFW1_strides = (((DLTensor*)SFW1_handle)[0].strides);
  void* SFW1 = (((DLTensor*)SFW1_handle)[0].data);
  if (!(((((DLTensor*)SFW3_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFW3 ndim mismatch, expected 2", (long long)((((DLTensor*)SFW3_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_SFW3_strides = (((DLTensor*)SFW3_handle)[0].strides);
  void* SFW3 = (((DLTensor*)SFW3_handle)[0].data);
  if (!(((((DLTensor*)Eid_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Eid ndim mismatch, expected 1", (long long)((((DLTensor*)Eid_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_Eid_strides = (((DLTensor*)Eid_handle)[0].strides);
  void* Eid = (((DLTensor*)Eid_handle)[0].data);
  if (!(((((((uint64_t)(((DLTensor*)A_handle)[0].dtype.bits)) * ((uint64_t)(((DLTensor*)A_handle)[0].dtype.lanes))) * ((uint64_t)((int64_t*)main_A_shape)[0])) * ((uint64_t)((int64_t*)main_A_shape)[1])) == (uint64_t)94371840))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A is a subtype, but total bits mismatch violates packed ABI constraint", (long long)((((((uint64_t)(((DLTensor*)A_handle)[0].dtype.bits)) * ((uint64_t)(((DLTensor*)A_handle)[0].dtype.lanes))) * ((uint64_t)((int64_t*)main_A_shape)[0])) * ((uint64_t)((int64_t*)main_A_shape)[1]))), (long long)((uint64_t)94371840));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_A_shape)[0]) == 4608))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_A_shape)[0])), (long long)(4608));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((int32_t)((int64_t*)main_A_shape)[1]) * 2) == 5120))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A shape[1] violates packed ABI constraint", (long long)((((int32_t)((int64_t*)main_A_shape)[1]) * 2)), (long long)(5120));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval;
  if ((main_A_strides == NULL)) {
    condval = 0;
  } else {
    condval = ((int32_t)((int64_t*)main_A_strides)[1]);
  }
  if (!((condval == 1))) {
    int32_t condval_1;
    if ((main_A_strides == NULL)) {
      condval_1 = 0;
    } else {
      condval_1 = ((int32_t)((int64_t*)main_A_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A strides[1] violates packed ABI constraint", (long long)(condval_1), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_2;
  if ((main_A_strides == NULL)) {
    condval_2 = 0;
  } else {
    condval_2 = ((int32_t)((int64_t*)main_A_strides)[0]);
  }
  if (!(((condval_2 * 2) == 5120))) {
    int32_t condval_3;
    if ((main_A_strides == NULL)) {
      condval_3 = 0;
    } else {
      condval_3 = ((int32_t)((int64_t*)main_A_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A strides[0] violates packed ABI constraint", (long long)((condval_3 * 2)), (long long)(5120));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)A_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)A_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)A_handle)[0].device.device_id) == dev_id))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A device_id violates packed ABI constraint", (long long)((((DLTensor*)A_handle)[0].device.device_id)), (long long)(dev_id));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)A_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A device_type mismatch, expected cuda", (long long)((((DLTensor*)A_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(A == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input A data pointer is NULL");
    return -1;
  }
  if (!((((((((uint64_t)(((DLTensor*)W1_handle)[0].dtype.bits)) * ((uint64_t)(((DLTensor*)W1_handle)[0].dtype.lanes))) * ((uint64_t)((int64_t*)main_W1_shape)[0])) * ((uint64_t)((int64_t*)main_W1_shape)[1])) * ((uint64_t)((int64_t*)main_W1_shape)[2])) == (uint64_t)2516582400))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W1 is a subtype, but total bits mismatch violates packed ABI constraint", (long long)(((((((uint64_t)(((DLTensor*)W1_handle)[0].dtype.bits)) * ((uint64_t)(((DLTensor*)W1_handle)[0].dtype.lanes))) * ((uint64_t)((int64_t*)main_W1_shape)[0])) * ((uint64_t)((int64_t*)main_W1_shape)[1])) * ((uint64_t)((int64_t*)main_W1_shape)[2]))), (long long)((uint64_t)2516582400));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_W1_shape)[0]) == 384))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W1 shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_W1_shape)[0])), (long long)(384));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_W1_shape)[1]) == 320))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W1 shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_W1_shape)[1])), (long long)(320));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((int32_t)((int64_t*)main_W1_shape)[2]) * 2) == 5120))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W1 shape[2] violates packed ABI constraint", (long long)((((int32_t)((int64_t*)main_W1_shape)[2]) * 2)), (long long)(5120));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_4;
  if ((main_W1_strides == NULL)) {
    condval_4 = 0;
  } else {
    condval_4 = ((int32_t)((int64_t*)main_W1_strides)[2]);
  }
  if (!((condval_4 == 1))) {
    int32_t condval_5;
    if ((main_W1_strides == NULL)) {
      condval_5 = 0;
    } else {
      condval_5 = ((int32_t)((int64_t*)main_W1_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W1 strides[2] violates packed ABI constraint", (long long)(condval_5), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_6;
  if ((main_W1_strides == NULL)) {
    condval_6 = 0;
  } else {
    condval_6 = ((int32_t)((int64_t*)main_W1_strides)[1]);
  }
  if (!(((condval_6 * 2) == 5120))) {
    int32_t condval_7;
    if ((main_W1_strides == NULL)) {
      condval_7 = 0;
    } else {
      condval_7 = ((int32_t)((int64_t*)main_W1_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W1 strides[1] violates packed ABI constraint", (long long)((condval_7 * 2)), (long long)(5120));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_8;
  if ((main_W1_strides == NULL)) {
    condval_8 = 0;
  } else {
    condval_8 = ((int32_t)((int64_t*)main_W1_strides)[0]);
  }
  if (!(((condval_8 * 2) == 1638400))) {
    int32_t condval_9;
    if ((main_W1_strides == NULL)) {
      condval_9 = 0;
    } else {
      condval_9 = ((int32_t)((int64_t*)main_W1_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W1 strides[0] violates packed ABI constraint", (long long)((condval_9 * 2)), (long long)(1638400));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)W1_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W1 byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)W1_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)W1_handle)[0].device.device_id) == dev_id))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W1 device_id violates packed ABI constraint", (long long)((((DLTensor*)W1_handle)[0].device.device_id)), (long long)(dev_id));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)W1_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W1 device_type mismatch, expected cuda", (long long)((((DLTensor*)W1_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(W1 == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input W1 data pointer is NULL");
    return -1;
  }
  if (!((((((((uint64_t)(((DLTensor*)W3_handle)[0].dtype.bits)) * ((uint64_t)(((DLTensor*)W3_handle)[0].dtype.lanes))) * ((uint64_t)((int64_t*)main_W3_shape)[0])) * ((uint64_t)((int64_t*)main_W3_shape)[1])) * ((uint64_t)((int64_t*)main_W3_shape)[2])) == (uint64_t)2516582400))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W3 is a subtype, but total bits mismatch violates packed ABI constraint", (long long)(((((((uint64_t)(((DLTensor*)W3_handle)[0].dtype.bits)) * ((uint64_t)(((DLTensor*)W3_handle)[0].dtype.lanes))) * ((uint64_t)((int64_t*)main_W3_shape)[0])) * ((uint64_t)((int64_t*)main_W3_shape)[1])) * ((uint64_t)((int64_t*)main_W3_shape)[2]))), (long long)((uint64_t)2516582400));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_W3_shape)[0]) == 384))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W3 shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_W3_shape)[0])), (long long)(384));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_W3_shape)[1]) == 320))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W3 shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_W3_shape)[1])), (long long)(320));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((int32_t)((int64_t*)main_W3_shape)[2]) * 2) == 5120))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W3 shape[2] violates packed ABI constraint", (long long)((((int32_t)((int64_t*)main_W3_shape)[2]) * 2)), (long long)(5120));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_10;
  if ((main_W3_strides == NULL)) {
    condval_10 = 0;
  } else {
    condval_10 = ((int32_t)((int64_t*)main_W3_strides)[2]);
  }
  if (!((condval_10 == 1))) {
    int32_t condval_11;
    if ((main_W3_strides == NULL)) {
      condval_11 = 0;
    } else {
      condval_11 = ((int32_t)((int64_t*)main_W3_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W3 strides[2] violates packed ABI constraint", (long long)(condval_11), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_12;
  if ((main_W3_strides == NULL)) {
    condval_12 = 0;
  } else {
    condval_12 = ((int32_t)((int64_t*)main_W3_strides)[1]);
  }
  if (!(((condval_12 * 2) == 5120))) {
    int32_t condval_13;
    if ((main_W3_strides == NULL)) {
      condval_13 = 0;
    } else {
      condval_13 = ((int32_t)((int64_t*)main_W3_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W3 strides[1] violates packed ABI constraint", (long long)((condval_13 * 2)), (long long)(5120));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_14;
  if ((main_W3_strides == NULL)) {
    condval_14 = 0;
  } else {
    condval_14 = ((int32_t)((int64_t*)main_W3_strides)[0]);
  }
  if (!(((condval_14 * 2) == 1638400))) {
    int32_t condval_15;
    if ((main_W3_strides == NULL)) {
      condval_15 = 0;
    } else {
      condval_15 = ((int32_t)((int64_t*)main_W3_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W3 strides[0] violates packed ABI constraint", (long long)((condval_15 * 2)), (long long)(1638400));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)W3_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W3 byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)W3_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)W3_handle)[0].device.device_id) == dev_id))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W3 device_id violates packed ABI constraint", (long long)((((DLTensor*)W3_handle)[0].device.device_id)), (long long)(dev_id));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)W3_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W3 device_type mismatch, expected cuda", (long long)((((DLTensor*)W3_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(W3 == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input W3 data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)SFA_handle)[0].dtype.code) == (uint8_t)1) && ((((DLTensor*)SFA_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)SFA_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input SFA dtype mismatch, expected uint32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_SFA_shape)[0]) == 184320))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFA shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_SFA_shape)[0])), (long long)(184320));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_16;
  if ((main_SFA_strides == NULL)) {
    condval_16 = 1;
  } else {
    condval_16 = ((int32_t)((int64_t*)main_SFA_strides)[0]);
  }
  if (!((condval_16 == 1))) {
    int32_t condval_17;
    if ((main_SFA_strides == NULL)) {
      condval_17 = 1;
    } else {
      condval_17 = ((int32_t)((int64_t*)main_SFA_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFA strides[0] violates packed ABI constraint", (long long)(condval_17), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)SFA_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFA byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)SFA_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)SFA_handle)[0].device.device_id) == dev_id))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFA device_id violates packed ABI constraint", (long long)((((DLTensor*)SFA_handle)[0].device.device_id)), (long long)(dev_id));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)SFA_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFA device_type mismatch, expected cuda", (long long)((((DLTensor*)SFA_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(SFA == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input SFA data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)SFW1_handle)[0].dtype.code) == (uint8_t)1) && ((((DLTensor*)SFW1_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)SFW1_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input SFW1 dtype mismatch, expected uint32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_SFW1_shape)[0]) == 384))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFW1 shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_SFW1_shape)[0])), (long long)(384));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_SFW1_shape)[1]) == 12800))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFW1 shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_SFW1_shape)[1])), (long long)(12800));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_18;
  if ((main_SFW1_strides == NULL)) {
    condval_18 = 1;
  } else {
    condval_18 = ((int32_t)((int64_t*)main_SFW1_strides)[1]);
  }
  if (!((condval_18 == 1))) {
    int32_t condval_19;
    if ((main_SFW1_strides == NULL)) {
      condval_19 = 1;
    } else {
      condval_19 = ((int32_t)((int64_t*)main_SFW1_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFW1 strides[1] violates packed ABI constraint", (long long)(condval_19), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_20;
  if ((main_SFW1_strides == NULL)) {
    condval_20 = 1;
  } else {
    condval_20 = ((int32_t)((int64_t*)main_SFW1_strides)[0]);
  }
  if (!((condval_20 == 12800))) {
    int32_t condval_21;
    if ((main_SFW1_strides == NULL)) {
      condval_21 = 1;
    } else {
      condval_21 = ((int32_t)((int64_t*)main_SFW1_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFW1 strides[0] violates packed ABI constraint", (long long)(condval_21), (long long)(12800));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)SFW1_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFW1 byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)SFW1_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)SFW1_handle)[0].device.device_id) == dev_id))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFW1 device_id violates packed ABI constraint", (long long)((((DLTensor*)SFW1_handle)[0].device.device_id)), (long long)(dev_id));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)SFW1_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFW1 device_type mismatch, expected cuda", (long long)((((DLTensor*)SFW1_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(SFW1 == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input SFW1 data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)SFW3_handle)[0].dtype.code) == (uint8_t)1) && ((((DLTensor*)SFW3_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)SFW3_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input SFW3 dtype mismatch, expected uint32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_SFW3_shape)[0]) == 384))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFW3 shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_SFW3_shape)[0])), (long long)(384));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_SFW3_shape)[1]) == 12800))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFW3 shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_SFW3_shape)[1])), (long long)(12800));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_22;
  if ((main_SFW3_strides == NULL)) {
    condval_22 = 1;
  } else {
    condval_22 = ((int32_t)((int64_t*)main_SFW3_strides)[1]);
  }
  if (!((condval_22 == 1))) {
    int32_t condval_23;
    if ((main_SFW3_strides == NULL)) {
      condval_23 = 1;
    } else {
      condval_23 = ((int32_t)((int64_t*)main_SFW3_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFW3 strides[1] violates packed ABI constraint", (long long)(condval_23), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_24;
  if ((main_SFW3_strides == NULL)) {
    condval_24 = 1;
  } else {
    condval_24 = ((int32_t)((int64_t*)main_SFW3_strides)[0]);
  }
  if (!((condval_24 == 12800))) {
    int32_t condval_25;
    if ((main_SFW3_strides == NULL)) {
      condval_25 = 1;
    } else {
      condval_25 = ((int32_t)((int64_t*)main_SFW3_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFW3 strides[0] violates packed ABI constraint", (long long)(condval_25), (long long)(12800));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)SFW3_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFW3 byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)SFW3_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)SFW3_handle)[0].device.device_id) == dev_id))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFW3 device_id violates packed ABI constraint", (long long)((((DLTensor*)SFW3_handle)[0].device.device_id)), (long long)(dev_id));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)SFW3_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SFW3 device_type mismatch, expected cuda", (long long)((((DLTensor*)SFW3_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(SFW3 == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input SFW3 data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)Eid_handle)[0].dtype.code) == (uint8_t)0) && ((((DLTensor*)Eid_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)Eid_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Eid dtype mismatch, expected int32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Eid_shape)[0]) == 36))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Eid shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Eid_shape)[0])), (long long)(36));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_26;
  if ((main_Eid_strides == NULL)) {
    condval_26 = 1;
  } else {
    condval_26 = ((int32_t)((int64_t*)main_Eid_strides)[0]);
  }
  if (!((condval_26 == 1))) {
    int32_t condval_27;
    if ((main_Eid_strides == NULL)) {
      condval_27 = 1;
    } else {
      condval_27 = ((int32_t)((int64_t*)main_Eid_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Eid strides[0] violates packed ABI constraint", (long long)(condval_27), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)Eid_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Eid byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)Eid_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Eid_handle)[0].device.device_id) == dev_id))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Eid device_id violates packed ABI constraint", (long long)((((DLTensor*)Eid_handle)[0].device.device_id)), (long long)(dev_id));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Eid_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Eid device_type mismatch, expected cuda", (long long)((((DLTensor*)Eid_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(Eid == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Eid data pointer is NULL");
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_1[1];
  void* callee_allocated_output_storage = stack_1;
  TVMFFIAny stack_2[1];
  void* callee_allocated_output_shape_storage = stack_2;
  TVMFFIAny stack_3[3];
  void* callee_allocated_output_prototype_storage = stack_3;
  (((int64_t*)callee_allocated_output_shape_storage)[0]) = (int64_t)4608;
  (((int64_t*)callee_allocated_output_shape_storage)[1]) = (int64_t)640;
  uint64_t v_ = (uint64_t)0;
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].data) = (*(void* *)(&(v_)));
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].shape) = (int64_t*)((void*)((char*)callee_allocated_output_shape_storage + (int64_t)0));
  uint64_t v__1 = (uint64_t)0;
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].strides) = (int64_t*)(*(void* *)(&(v__1)));
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].ndim) = 2;
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].dtype.code) = (uint8_t)2;
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].dtype.bits) = (uint8_t)32;
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].dtype.lanes) = (uint16_t)1;
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].byte_offset) = (uint64_t)0;
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].device.device_id) = dev_id;
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].device.device_type) = (DLDeviceType)2;
  int32_t C_allocation_status = TVMFFIEnvTensorAlloc((((DLTensor*)callee_allocated_output_prototype_storage) + 0), ((void*)((char*)callee_allocated_output_storage + (int64_t)8)));
  if (C_allocation_status != 0) {
    return C_allocation_status;
  }
  void* C_handle = ((void*)((char*)(((TVMFFIAny*)callee_allocated_output_storage)[0].v_ptr) + (int64_t)24));
  bool main_C_is_null = (C_handle == NULL);
  if (!(!main_C_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.C is expected to have non-NULL pointer");
    return -1;
  }
  void* main_C_shape = (((DLTensor*)C_handle)[0].shape);
  if (!(((((DLTensor*)C_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C ndim mismatch, expected 2", (long long)((((DLTensor*)C_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_C_strides = (((DLTensor*)C_handle)[0].strides);
  void* C = (((DLTensor*)C_handle)[0].data);
  if (!(((((((DLTensor*)C_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)C_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)C_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input C dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_C_shape)[0]) == 4608))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_C_shape)[0])), (long long)(4608));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_C_shape)[1]) == 640))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_C_shape)[1])), (long long)(640));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_28;
  if ((main_C_strides == NULL)) {
    condval_28 = 1;
  } else {
    condval_28 = ((int32_t)((int64_t*)main_C_strides)[1]);
  }
  if (!((condval_28 == 1))) {
    int32_t condval_29;
    if ((main_C_strides == NULL)) {
      condval_29 = 1;
    } else {
      condval_29 = ((int32_t)((int64_t*)main_C_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C strides[1] violates packed ABI constraint", (long long)(condval_29), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_30;
  if ((main_C_strides == NULL)) {
    condval_30 = 1;
  } else {
    condval_30 = ((int32_t)((int64_t*)main_C_strides)[0]);
  }
  if (!((condval_30 == 640))) {
    int32_t condval_31;
    if ((main_C_strides == NULL)) {
      condval_31 = 1;
    } else {
      condval_31 = ((int32_t)((int64_t*)main_C_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C strides[0] violates packed ABI constraint", (long long)(condval_31), (long long)(640));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)C_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)C_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)C_handle)[0].device.device_id) == dev_id))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C device_id violates packed ABI constraint", (long long)((((DLTensor*)C_handle)[0].device.device_id)), (long long)(dev_id));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)C_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C device_type mismatch, expected cuda", (long long)((((DLTensor*)C_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(C == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input C data pointer is NULL");
    return -1;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)dev_id);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
  if (__tvm_set_device_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_set_device", &__tvm_set_device_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_1;
  result_1.type_index = kTVMFFINone;
  result_1.zero_padding = 0;
  result_1.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_set_device_packed, (TVMFFIAny*) stack_ffi_any, 2, &result_1) != 0) {
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_4[16];
  void* A_desc = stack_4;
  if (A_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = A_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)14);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)2);
  if (A == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = A;
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)5120);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)4608);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)2560);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)3);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_int64) = (int64_t)0;
  if (__tvm_tensormap_create_tiled_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_tensormap_create_tiled", &__tvm_tensormap_create_tiled_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_2;
  result_2.type_index = kTVMFFINone;
  result_2.zero_padding = 0;
  result_2.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed, (TVMFFIAny*) stack_ffi_any, 16, &result_2) != 0) {
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_5[16];
  void* W1_desc = stack_5;
  if (W1_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = W1_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)14);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)3);
  if (W1 == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = W1;
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)5120);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)320);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)384);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)2560);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)819200);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = ((int64_t)64);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[16].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[17].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[17].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[17].v_int64) = ((int64_t)3);
  (((TVMFFIAny*)stack_ffi_any)[18].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[18].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[18].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[19].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[19].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[19].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[20].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[20].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[20].v_int64) = (int64_t)0;
  if (__tvm_tensormap_create_tiled_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_tensormap_create_tiled", &__tvm_tensormap_create_tiled_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_3;
  result_3.type_index = kTVMFFINone;
  result_3.zero_padding = 0;
  result_3.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed, (TVMFFIAny*) stack_ffi_any, 20, &result_3) != 0) {
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_6[16];
  void* W3_desc = stack_6;
  if (W3_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = W3_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)14);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)3);
  if (W3 == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = W3;
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)5120);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)320);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)384);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)2560);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)819200);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = ((int64_t)64);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[16].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[17].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[17].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[17].v_int64) = ((int64_t)3);
  (((TVMFFIAny*)stack_ffi_any)[18].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[18].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[18].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[19].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[19].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[19].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[20].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[20].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[20].v_int64) = (int64_t)0;
  if (__tvm_tensormap_create_tiled_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_tensormap_create_tiled", &__tvm_tensormap_create_tiled_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_4;
  result_4.type_index = kTVMFFINone;
  result_4.zero_padding = 0;
  result_4.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed, (TVMFFIAny*) stack_ffi_any, 20, &result_4) != 0) {
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_7[16];
  void* SFW1_desc = stack_7;
  if (SFW1_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = SFW1_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)2);
  if (SFW1 == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = SFW1;
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)12800);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)384);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)4);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)51200);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)64);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_int64) = (int64_t)0;
  if (__tvm_tensormap_create_tiled_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_tensormap_create_tiled", &__tvm_tensormap_create_tiled_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_5;
  result_5.type_index = kTVMFFINone;
  result_5.zero_padding = 0;
  result_5.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed, (TVMFFIAny*) stack_ffi_any, 16, &result_5) != 0) {
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_8[16];
  void* SFW3_desc = stack_8;
  if (SFW3_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = SFW3_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)2);
  if (SFW3 == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = SFW3;
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)12800);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)384);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)4);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)51200);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)64);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_int64) = (int64_t)0;
  if (__tvm_tensormap_create_tiled_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_tensormap_create_tiled", &__tvm_tensormap_create_tiled_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_6;
  result_6.type_index = kTVMFFINone;
  result_6.zero_padding = 0;
  result_6.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed, (TVMFFIAny*) stack_ffi_any, 16, &result_6) != 0) {
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_9[16];
  void* C_desc = stack_9;
  if (C_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = C_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)7);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)2);
  if (C == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = C;
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)640);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)4608);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)4);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)2560);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)32);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)3);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_int64) = (int64_t)0;
  if (__tvm_tensormap_create_tiled_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_tensormap_create_tiled", &__tvm_tensormap_create_tiled_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_7;
  result_7.type_index = kTVMFFINone;
  result_7.zero_padding = 0;
  result_7.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed, (TVMFFIAny*) stack_ffi_any, 16, &result_7) != 0) {
    return -1;
  }
  if (A_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = A_desc;
  if (C_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = C_desc;
  if (Eid == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = Eid;
  if (SFA == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = SFA;
  if (SFW1_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_ptr) = SFW1_desc;
  if (SFW3_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_ptr) = SFW3_desc;
  if (W1_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_ptr) = W1_desc;
  if (W3_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_ptr) = W3_desc;
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)5);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)36);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)202752);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = (int64_t)0;
  if (main_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "main_kernel", &main_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_8;
  result_8.type_index = kTVMFFINone;
  result_8.zero_padding = 0;
  result_8.v_int64 = 0;
  if (TVMFFIFunctionCall(main_kernel_packed, (TVMFFIAny*) stack_ffi_any, 14, &result_8) != 0) {
    return -1;
  }
  (((TVMFFIAny*)result)[0].type_index) = 70;
  (((TVMFFIAny*)result)[0].zero_padding) = 0;
  (((TVMFFIAny*)result)[0].v_int64) = 0;
  (((TVMFFIAny*)result)[0].v_ptr) = (((TVMFFIAny*)callee_allocated_output_storage)[0].v_ptr);
  return 0;
}

