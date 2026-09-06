/*
 * Copyright 2026 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/kernels/neutron/neutron.h"

tflite::MicroOpResolver &MODEL_GetOpsResolver()
{
    static tflite::MicroMutableOpResolver<26> s_microOpResolver;

    s_microOpResolver.AddPad();
    s_microOpResolver.AddSoftmax();
    s_microOpResolver.AddDequantize();
    s_microOpResolver.AddFullyConnected();
    s_microOpResolver.AddMean();
    s_microOpResolver.AddNeg();
    s_microOpResolver.AddQuantize();
    s_microOpResolver.AddAdd();
    s_microOpResolver.AddRsqrt();
    s_microOpResolver.AddMul();
    s_microOpResolver.AddTanh();
    s_microOpResolver.AddLogistic();
    s_microOpResolver.AddConcatenation();
    s_microOpResolver.AddSquaredDifference();
    s_microOpResolver.AddShape();
    s_microOpResolver.AddStridedSlice();
    s_microOpResolver.AddPack();
    s_microOpResolver.AddFill();
    s_microOpResolver.AddTranspose();
    s_microOpResolver.AddUnpack();
    s_microOpResolver.AddSplit();
    s_microOpResolver.AddSub();
    s_microOpResolver.AddCustom(tflite::GetString_NEUTRON_GRAPH(), tflite::Register_NEUTRON_GRAPH());

    return s_microOpResolver;
}

