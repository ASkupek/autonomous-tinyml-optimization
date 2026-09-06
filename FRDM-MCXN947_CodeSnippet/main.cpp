/*
 * Copyright 2020-2022 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

 // NOTE: This is just a demo code at the moment. We will update it and make it cleaer in next versions!

#include "board_init.h"
#include "demo_config.h"
#include "demo_info.h"
#include "fsl_debug_console.h"
#include "image.h"
#include "image_utils.h"
#include "model.h"
#include "output_postproc.h"
#include "timer.h"
#include "fsl_lpuart.h"

#define DEMO_LPUART LPUART4
#define DEMO_LPUART_CLK_FREQ CLOCK_GetLPFlexCommClkFreq(4u)

#define WINDOW_SIZE 10
#define NUM_INPUTS 4
#define NUM_OUTPUTS 1
#define DEMO_LPUART            LPUART4
#define DEMO_LPUART_CLK_FREQ   CLOCK_GetLPFlexCommClkFreq(4u)

#define WINDOW_SIZE            10
#define NUM_FEATURES           4
#define NUM_OUTPUTS            1

#define PACKET_HEADER_0        0xAA
#define PACKET_HEADER_1        0xBB


int main(void)
{
    BOARD_Init();
    TIMER_Init();

    /* ============================================================
     * UART INIT
     * ============================================================ */

    lpuart_config_t config;

    LPUART_GetDefaultConfig(&config);

    config.baudRate_Bps = 115200U;
    config.enableTx = true;
    config.enableRx = true;

    LPUART_Init(
        DEMO_LPUART,
        &config,
        DEMO_LPUART_CLK_FREQ
    );


    /* ============================================================
     * MODEL INIT
     * ============================================================ */

    if (MODEL_Init() != kStatus_Success)
    {
        PRINTF("Failed initializing model" EOL);

        while (1)
        {
        }
    }


    /* ============================================================
     * GET MODEL INPUT
     * ============================================================ */

    tensor_dims_t inputDims;
    tensor_type_t inputType;

    float *modelInputData =
        reinterpret_cast<float *>(
            MODEL_GetInputTensorData(
                &inputDims,
                &inputType
            )
        );


    /* ============================================================
     * GET MODEL OUTPUT
     * ============================================================ */

    tensor_dims_t outputDims;
    tensor_type_t outputType;

    float *modelOutputData =
        reinterpret_cast<float *>(
            MODEL_GetOutputTensorData(
                &outputDims,
                &outputType
            )
        );


    /* ============================================================
     * DEBUG MODEL INFORMATION
     * ============================================================ */

    PRINTF("Model initialized" EOL);

    PRINTF(
        "Input type: %d" EOL,
        inputType
    );

    PRINTF(
        "Output type: %d" EOL,
        outputType
    );


    /* ============================================================
     * RING BUFFER
     *
     * inputWindow:
     *
     *        [10][4]
     *
     * Each row:
     *
     *        speed
     *        acceleration
     *        lane_offset
     *        heading_error
     *
     * ============================================================ */

    float inputWindow[WINDOW_SIZE][NUM_FEATURES] = {0.0f};

    float currentInput[NUM_FEATURES] = {0.0f};

    /*
     * nextIndex points to the position where the next
     * timestep will be written.
     *
     * Example:
     *
     * nextIndex = 0
     *
     * write t0 -> [0]
     * nextIndex = 1
     *
     * ...
     *
     * write t9 -> [9]
     * nextIndex = 0
     *
     * next write overwrites the oldest timestep.
     */
    int nextIndex = 0;

    int windowCount = 0;


    /* ============================================================
     * MAIN LOOP
     * ============================================================ */

    while (1)
    {
        uint8_t header[2] = {0, 0};


        /* ========================================================
         * WAIT FOR PACKET HEADER
         *
         * Packet:
         *
         *     AA BB
         *     float
         *     float
         *     float
         *     float
         *
         * ======================================================== */

        while (1)
        {
            LPUART_ReadBlocking(
                DEMO_LPUART,
                &header[0],
                1
            );

            if (header[0] == PACKET_HEADER_0)
            {
                LPUART_ReadBlocking(
                    DEMO_LPUART,
                    &header[1],
                    1
                );

                if (header[1] == PACKET_HEADER_1)
                {
                    break;
                }
            }
        }


        /* ========================================================
         * READ ONE TIMESTEP
         *
         * currentInput:
         *
         *     [speed,
         *      acceleration,
         *      lane_offset,
         *      heading_error]
         *
         * These values are ALREADY SCALED by Python.
         * ======================================================== */

        LPUART_ReadBlocking(
            DEMO_LPUART,
            reinterpret_cast<uint8_t *>(currentInput),
            NUM_FEATURES * sizeof(float)
        );


        /* ========================================================
         * ADD CURRENT TIMESTEP TO RING BUFFER
         * ======================================================== */

        for (int f = 0; f < NUM_FEATURES; f++)
        {
            inputWindow[nextIndex][f] = currentInput[f];
        }


        /* ========================================================
         * UPDATE RING BUFFER INDEX
         * ======================================================== */

        nextIndex++;

        if (nextIndex >= WINDOW_SIZE)
        {
            nextIndex = 0;
        }


        /* ========================================================
         * UPDATE WINDOW COUNT
         * ======================================================== */

        if (windowCount < WINDOW_SIZE)
        {
            windowCount++;
        }


        /* ========================================================
         * WINDOW NOT FULL YET
         *
         * Need 10 timesteps before running GRU.
         * ======================================================== */

        if (windowCount < WINDOW_SIZE)
        {
            float steering = 0.0f;

            LPUART_WriteBlocking(
                DEMO_LPUART,
                reinterpret_cast<const uint8_t *>(&steering),
                sizeof(float)
            );

            continue;
        }


        /* ========================================================
         * COPY RING BUFFER -> TFLITE INPUT
         *
         * IMPORTANT:
         *
         * Ring buffer is physically:
         *
         *     newest / oldest positions depend on nextIndex
         *
         * We need to provide the model with:
         *
         *     oldest
         *     ...
         *     newest
         *
         * So start at nextIndex.
         *
         * ======================================================== */

        for (int t = 0; t < WINDOW_SIZE; t++)
        {
            int bufferIndex = nextIndex + t;

            if (bufferIndex >= WINDOW_SIZE)
            {
                bufferIndex -= WINDOW_SIZE;
            }

            for (int f = 0; f < NUM_FEATURES; f++)
            {
                modelInputData[t * NUM_FEATURES + f] =inputWindow[bufferIndex][f];
            }
        }


        /* ========================================================
         * RUN MODEL
         * ======================================================== */

        MODEL_RunInference();


        /* ========================================================
         * SEND MODEL OUTPUT
         * ======================================================== */

        LPUART_WriteBlocking(
            DEMO_LPUART,
            reinterpret_cast<const uint8_t *>(modelOutputData),
            NUM_OUTPUTS * sizeof(float)
        );
    }
}