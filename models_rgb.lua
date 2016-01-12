require 'torch'
require 'nn'
require 'dpnn'
require 'cudnn'

local models = {}

function models.create_G32x32(dimensions, cuda)
    local model = nn.Sequential()

    model:add(nn.JoinTable(2, 2))

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    local inner = nn.Sequential()

    inner:add(cudnn.SpatialConvolution(1+1, 16, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(nn.SpatialBatchNormalization(16))
    inner:add(cudnn.ReLU(true))

    inner:add(cudnn.SpatialConvolution(16, 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(nn.SpatialBatchNormalization(32))
    inner:add(cudnn.ReLU(true))

    inner:add(cudnn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(nn.SpatialBatchNormalization(64))
    inner:add(cudnn.ReLU(true))

    inner:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(nn.SpatialBatchNormalization(128))
    inner:add(cudnn.ReLU(true))

    inner:add(cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(nn.SpatialBatchNormalization(64))
    inner:add(cudnn.ReLU(true))

    inner:add(cudnn.SpatialConvolution(64, 3, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(nn.Sigmoid())

    model:add(inner)

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        inner:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_G32x32b(dimensions, cuda)
    local model = nn.Sequential()

    model:add(nn.JoinTable(2, 2))

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    local inner = nn.Sequential()

    inner:add(cudnn.SpatialConvolution(1+1, 16, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(nn.SpatialBatchNormalization(16))
    inner:add(cudnn.ReLU(true))

    inner:add(cudnn.SpatialConvolution(16, 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(nn.SpatialBatchNormalization(32))
    inner:add(cudnn.ReLU(true))
    inner:add(nn.SpatialMaxPooling(2, 2))

    inner:add(cudnn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(nn.SpatialBatchNormalization(64))
    inner:add(cudnn.ReLU(true))
    inner:add(nn.SpatialMaxPooling(2, 2))

    inner:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(nn.SpatialBatchNormalization(128))
    inner:add(cudnn.ReLU(true))

    inner:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(nn.SpatialBatchNormalization(256))
    inner:add(cudnn.ReLU(true))

    inner:add(nn.SpatialUpSamplingNearest(2))
    inner:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(nn.SpatialBatchNormalization(128))
    inner:add(cudnn.ReLU(true))

    inner:add(nn.SpatialUpSamplingNearest(2))
    inner:add(cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(nn.SpatialBatchNormalization(64))
    inner:add(cudnn.ReLU(true))

    inner:add(cudnn.SpatialConvolution(64, 3, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(nn.Sigmoid())

    model:add(inner)

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        inner:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_G32x32c(dimensions, cuda)
    local model = nn.Sequential()

    model:add(nn.JoinTable(2, 2))

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    local inner = nn.Sequential()
    local conc = nn.Concat(2)
    local left = nn.Sequential()
    local right = nn.Sequential()

    left:add(nn.Identity())

    right:add(cudnn.SpatialConvolution(1+1, 16, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    right:add(nn.SpatialBatchNormalization(16))
    right:add(cudnn.ReLU(true))

    right:add(cudnn.SpatialConvolution(16, 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    right:add(nn.SpatialBatchNormalization(32))
    right:add(cudnn.ReLU(true))
    right:add(nn.SpatialMaxPooling(2, 2))

    right:add(cudnn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    right:add(nn.SpatialBatchNormalization(64))
    right:add(cudnn.ReLU(true))
    right:add(nn.SpatialMaxPooling(2, 2))

    right:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    right:add(nn.SpatialBatchNormalization(128))
    right:add(cudnn.ReLU(true))

    right:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    right:add(nn.SpatialBatchNormalization(256))
    right:add(cudnn.ReLU(true))

    right:add(nn.SpatialUpSamplingNearest(2))
    right:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    right:add(nn.SpatialBatchNormalization(128))
    right:add(cudnn.ReLU(true))

    right:add(nn.SpatialUpSamplingNearest(2))
    right:add(cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    right:add(nn.SpatialBatchNormalization(64))
    right:add(cudnn.ReLU(true))

    conc:add(left)
    conc:add(right)
    inner:add(conc)

    inner:add(cudnn.SpatialConvolution(2+64, 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(nn.SpatialBatchNormalization(32))
    inner:add(cudnn.ReLU(true))

    inner:add(cudnn.SpatialConvolution(32, 3, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(nn.Sigmoid())

    model:add(inner)

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        inner:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

-- Creates G, which is identical to the decoder part of G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G(dimensions, cuda)
    return models.create_G32x32c(dimensions, cuda)
end

-- Creates D.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_D(dimensions, cuda)
    return models.create_D32x32b(dimensions, cuda)
end

function models.create_D32x32(dimensions, cuda)
    local model = nn.Sequential()

    --model:add(nn.CAddTable())
    model:add(nn.JoinTable(2, 2))

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    local inner = nn.Sequential()

    -- 32x32
    inner:add(nn.SpatialConvolution(3+1, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(cudnn.ReLU(true))
    inner:add(nn.SpatialDropout(0.25))
    inner:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 32x32
    inner:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(cudnn.ReLU(true))
    inner:add(nn.SpatialDropout(0.25))
    --inner:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 16x16
    inner:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(cudnn.ReLU(true))
    inner:add(nn.SpatialDropout(0.25))
    inner:add(nn.SpatialMaxPooling(2, 2))

    -- 8x8
    inner:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(cudnn.ReLU(true))
    inner:add(nn.SpatialDropout(0.5))
    inner:add(nn.SpatialMaxPooling(2, 2))

    local height = dimensions[2] * 0.5 * 0.5 * 0.5
    local width = dimensions[3] * 0.5 * 0.5 * 0.5
    -- 4x4
    inner:add(nn.View(256*height*width))
    inner:add(nn.Linear(256*height*width, 128))
    inner:add(nn.PReLU())
    inner:add(nn.Dropout(0.5))
    inner:add(nn.Linear(128, 1))
    inner:add(nn.Sigmoid())

    model:add(inner)

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        inner:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_D32x32b(dimensions, cuda)
    local model = nn.Sequential()

    --model:add(nn.CAddTable())
    model:add(nn.JoinTable(2, 2))

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    local inner = nn.Sequential()

    -- 32x32
    inner:add(nn.SpatialConvolution(3+1, 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(cudnn.ReLU(true))
    inner:add(nn.SpatialDropout(0.25))
    inner:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 32x32
    inner:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(cudnn.ReLU(true))
    inner:add(nn.SpatialDropout(0.25))
    --inner:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 16x16
    inner:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(cudnn.ReLU(true))
    inner:add(nn.SpatialDropout(0.25))
    inner:add(nn.SpatialMaxPooling(2, 2))

    -- 8x8
    inner:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    inner:add(cudnn.ReLU(true))
    inner:add(nn.SpatialDropout(0.5))
    inner:add(nn.SpatialMaxPooling(2, 2))

    local height = dimensions[2] * 0.5 * 0.5 * 0.5
    local width = dimensions[3] * 0.5 * 0.5 * 0.5
    -- 4x4
    inner:add(nn.View(128*height*width))
    inner:add(nn.Linear(128*height*width, 128))
    inner:add(nn.PReLU())
    inner:add(nn.Dropout(0.5))
    inner:add(nn.Linear(128, 1))
    inner:add(nn.Sigmoid())

    model:add(inner)

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        inner:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

return models
