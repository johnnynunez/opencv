// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUDNN_RECURRENT_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUDNN_RECURRENT_HPP

#include "cudnn.hpp"
#include <cudnn.h>


namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cudnn {

/**
 */
class DropoutDescriptor
{
public:
    DropoutDescriptor() noexcept = default;
    DropoutDescriptor(const DropoutDescriptor &) = delete;
    DropoutDescriptor(DropoutDescriptor &&other) noexcept : descriptor{other.descriptor}
    {
        states = std::move(other.states);
        other.descriptor = nullptr;
    }

    /**
     */
    DropoutDescriptor(const Handle &handle, float dropout)
    {
        CUDA4DNN_CHECK_CUDNN(cudnnCreateDropoutDescriptor(&descriptor));

        // we need additional memory for dropout descriptor
        size_t stateSize;
        CUDA4DNN_CHECK_CUDNN(cudnnDropoutGetStatesSize(handle.get(), &stateSize));
        states.reset(stateSize);

        try
        {
            auto seed = 1234ull; // Pick a seed.
            CUDA4DNN_CHECK_CUDNN(cudnnSetDropoutDescriptor(descriptor, handle.get(), dropout,
                                                           states.get().get(), stateSize, seed));
        }
        catch (...)
        {
            CUDA4DNN_CHECK_CUDNN(cudnnDestroyDropoutDescriptor(descriptor));
            throw;
        }
    }

    ~DropoutDescriptor() noexcept
    {
        if (descriptor)
        {
            CUDA4DNN_CHECK_CUDNN(cudnnDestroyDropoutDescriptor(descriptor));
        }
    }

    DropoutDescriptor &operator=(const DropoutDescriptor &) = delete;
    DropoutDescriptor &operator=(DropoutDescriptor &&other) noexcept
    {
        descriptor = other.descriptor;
        states = std::move(other.states);
        other.descriptor = nullptr;
        return *this;
    };

    cudnnDropoutDescriptor_t get() const noexcept { return descriptor; }

private:
    cudnnDropoutDescriptor_t descriptor{nullptr};

    using value_type = typename ManagedPtr<char>::element_type;
    ManagedPtr<value_type> states;
};

/**
 */
template<class T>
class RNNDescriptor
{
public:
    RNNDescriptor() noexcept = default;
    RNNDescriptor(const RNNDescriptor &) = delete;
    RNNDescriptor(RNNDescriptor &&other) noexcept : descriptor{other.descriptor}
    {
        other.descriptor = nullptr;
    }

    RNNDescriptor(const Handle &handle, RNNMode mode, int hidden_size, int num_layers,
                  bool bidirectional, const DropoutDescriptor &dropoutDesc, int input_size, int proj_size, uint32_t aux_flags)
    {
        CUDA4DNN_CHECK_CUDNN(cudnnCreateRNNDescriptor(&descriptor));

        cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;
        cudnnRNNMode_t cellMode = [mode] {
            switch (mode)
            {
                case RNNMode::RNN_RELU:
                    return CUDNN_RNN_RELU;
                case RNNMode::RNN_TANH:
                    return CUDNN_RNN_TANH;
                case RNNMode::LSTM:
                    return CUDNN_LSTM;
                case RNNMode::GRU:
                    return CUDNN_GRU;
                default:
                    return CUDNN_LSTM;
            }
        }();

        cudnnDirectionMode_t dirMode = bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
        cudnnRNNInputMode_t inputMode = CUDNN_LINEAR_INPUT;
        cudnnDataType_t dataType = detail::get_data_type<T>();
        cudnnMathType_t mathType = CUDNN_DEFAULT_MATH;
        cudnnRNNBiasMode_t biasMode = CUDNN_RNN_DOUBLE_BIAS;

        try
        {
            CUDA4DNN_CHECK_CUDNN(cudnnSetRNNDescriptor_v8(
                    descriptor,
                    algo,
                    cellMode,
                    biasMode,
                    dirMode,
                    inputMode,
                    dataType,
                    CUDNN_DATA_FLOAT, // Assuming FP32 computation
                    mathType,
                    input_size,
                    hidden_size,
                    proj_size,
                    num_layers,
                    dropoutDesc.get(),
                    aux_flags
            ));
        }
        catch (...)
        {
            CUDA4DNN_CHECK_CUDNN(cudnnDestroyRNNDescriptor(descriptor));
            throw;
        }
    }

    ~RNNDescriptor() noexcept
    {
        if (descriptor)
        {
            CUDA4DNN_CHECK_CUDNN(cudnnDestroyRNNDescriptor(descriptor));
        }
    }

    RNNDescriptor &operator=(const RNNDescriptor &) = delete;
    RNNDescriptor &operator=(RNNDescriptor &&other) noexcept
    {
        descriptor = other.descriptor;
        other.descriptor = nullptr;
        return *this;
    };

    cudnnRNNDescriptor_t get() const noexcept { return descriptor; }

private:
    cudnnRNNDescriptor_t descriptor{nullptr};
};


template <class T>
void LSTMForward(const Handle &handle, const RNNDescriptor<T> &rnnDesc,
                 cudnnRNNDataDescriptor_t xDesc, DevicePtr<const T> x,
                 cudnnRNNDataDescriptor_t yDesc, DevicePtr<T> y,
                 cudnnTensorDescriptor_t hDesc, DevicePtr<const T> hx,
                 DevicePtr<T> hy, cudnnTensorDescriptor_t cDesc,
                 DevicePtr<const T> cx, DevicePtr<T> cy, size_t weightSpaceSize,
                 DevicePtr<const T> weightSpace, WorkspaceInstance workspace,
                 size_t reserveSpaceSize, DevicePtr<T> reserveSpace) {
  CV_Assert(handle);
  CUDA4DNN_CHECK_CUDNN(cudnnRNNForward(
      handle.get(), rnnDesc.get(), CUDNN_FWD_MODE_INFERENCE,
      nullptr, // docs say use this as null on >= 8.9.7
      xDesc, x.get(), yDesc, y.get(), hDesc, hx.get(), hy.get(), cDesc,
      cx.get(), cy.get(), weightSpaceSize, weightSpace.get(),
      workspace.size_in_bytes(), workspace.get().get(), reserveSpaceSize,
      reserveSpace.get()));
}

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif //OPENCV_DNN_CUDA4DNN_CSL_CUDNN_RECURRENT_HPP
