require 'torch'
require 'optim'
require 'pl'
require 'image'

local adversarial = {}

-- this variable will save the accuracy values of D
adversarial.accs = {}

-- function to calculate the mean of a list of numbers
function adversarial.mean(t)
    local sum = 0
    local count = 0

    for k,v in pairs(t) do
        if type(v) == 'number' then
            sum = sum + v
            count = count + 1
        end
    end

    return (sum / count)
end

-- main training function
function adversarial.train(trainData, maxAccuracyD, accsInterval)
    EPOCH = EPOCH or 1
    local N_epoch = OPT.N_epoch
    if N_epoch <= 0 then
        N_epoch = 100
    end
    local dataBatchSize = OPT.batchSize / 2 -- size of a half-batch for D or G
    local time = sys.clock()

    -- variables to track D's accuracy and adjust learning rates
    local lastAccuracyD = 0.0
    local doTrainD = true
    local countTrainedD = 0
    local countNotTrainedD = 0

    samples = nil
    local batchIdx = 0

    -- do one epoch
    -- While this function is structured like one that picks example batches in consecutive order,
    -- in reality the examples (per batch) will be picked randomly
    print(string.format("<trainer> Epoch #%d [batchSize = %d]", EPOCH, OPT.batchSize))
    for batchIdx=1,N_epoch do
        -- size of this batch, will usually be dataBatchSize but can be lower at the end
        --local thisBatchSize = math.min(OPT.batchSize, N_epoch - t + 1)

        -- Inputs for D, either original or generated images
        local inputs = torch.Tensor(OPT.batchSize, 3, IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])

        -- target y-values
        local targets = torch.Tensor(OPT.batchSize)

        -- tensor to use for noise for G
        --local noiseInputs = torch.Tensor(thisBatchSize, OPT.noiseDim)
        local noiseInputs = torch.Tensor(OPT.batchSize, NOISE_DIM[1], NOISE_DIM[2], NOISE_DIM[3])
        local condInputs = torch.Tensor(OPT.batchSize, COND_DIM[1], COND_DIM[2], COND_DIM[3])

        ----------------------------------------------------------------------
        -- create closure to evaluate f(X) and df/dX of D
        local fevalD = function(x)
            collectgarbage()
            local confusion_batch_D = optim.ConfusionMatrix(CLASSES)
            confusion_batch_D:zero()

            if x ~= PARAMETERS_D then -- get new parameters
                PARAMETERS_D:copy(x)
            end

            GRAD_PARAMETERS_D:zero() -- reset gradients

            --  forward pass
            -- condInputs = y, inputs = uv
            local outputs = MODEL_D:forward({condInputs, inputs})
            local f = CRITERION:forward(outputs, targets)

            -- backward pass
            local df_do = CRITERION:backward(outputs, targets)
            MODEL_D:backward({condInputs, inputs}, df_do)

            -- penalties (L1 and L2):
            if OPT.D_L1 ~= 0 or OPT.D_L2 ~= 0 then
                -- Loss:
                f = f + OPT.D_L1 * torch.norm(PARAMETERS_D, 1)
                f = f + OPT.D_L2 * torch.norm(PARAMETERS_D, 2)^2/2
                -- Gradients:
                GRAD_PARAMETERS_D:add(torch.sign(PARAMETERS_D):mul(OPT.D_L1) + PARAMETERS_D:clone():mul(OPT.D_L2) )
            end

            -- update confusion (add 1 since targets are binary)
            for i=1,OPT.batchSize do
                local c
                if outputs[i][1] > 0.5 then c = 2 else c = 1 end
                CONFUSION:add(c, targets[i]+1)
                confusion_batch_D:add(c, targets[i]+1)
            end

            -- Clamp D's gradients
            -- This helps a bit against D suddenly giving up (only outputting y=1 or y=0)
            if OPT.D_clamp ~= 0 then
                GRAD_PARAMETERS_D:clamp((-1)*OPT.D_clamp, OPT.D_clamp)
            end

            -- Calculate accuracy of D on this batch
            confusion_batch_D:updateValids()
            local tV = confusion_batch_D.totalValid

            -- Add this batch's accuracy to the history of D's accuracies
            -- Also, keep that history to a fixed size
            adversarial.accs[#adversarial.accs+1] = tV
            if #adversarial.accs > accsInterval then
                table.remove(adversarial.accs, 1)
            end

            -- Mean accuracy of D over the last couple of batches
            local accAvg = adversarial.mean(adversarial.accs)

            -- We will only train D if its mean accuracy over the last couple of batches
            -- was below the defined maximum (maxAccuracyD). This protects a bit against
            -- G generating garbage.
            doTrainD = (accAvg < maxAccuracyD)
            lastAccuracyD = tV
            if doTrainD then
                countTrainedD = countTrainedD + 1
                return f,GRAD_PARAMETERS_D
            else
                countNotTrainedD = countNotTrainedD + 1

                -- The interruptable* Optimizers dont train when false is returned
                -- Maybe that would be equivalent to just returning 0 for all gradients?
                return false,false
            end
        end

        ----------------------------------------------------------------------
        -- create closure to evaluate f(X) and df/dX of generator
        local fevalG_on_D = function(x)
            collectgarbage()
            if x ~= PARAMETERS_G then -- get new parameters
                PARAMETERS_G:copy(x)
            end

            GRAD_PARAMETERS_G:zero() -- reset gradients

            -- forward pass
            --local samples = NN_UTILS.createImagesFromNoise(noiseInputs, false, true)
            local samples = MODEL_G:forward({noiseInputs, condInputs})
            -- condInputs = y, samples = uv
            local outputs = MODEL_D:forward({condInputs, samples})
            local f = CRITERION:forward(outputs, targets)

            --  backward pass
            local df_samples = CRITERION:backward(outputs, targets)
            MODEL_D:backward({condInputs, samples}, df_samples)
            local df_do = MODEL_D.modules[1].gradInput[2] -- 1=grad of y/condInput, 2=grad of uv/samples
            MODEL_G:backward({noiseInputs, condInputs}, df_do)

            -- penalties (L1 and L2):
            if OPT.G_L1 ~= 0 or OPT.G_L2 ~= 0 then
                -- Loss:
                f = f + OPT.G_L1 * torch.norm(PARAMETERS_G, 1)
                f = f + OPT.G_L2 * torch.norm(PARAMETERS_G, 2)^2/2
                -- Gradients:
                GRAD_PARAMETERS_G:add(torch.sign(PARAMETERS_G):mul(OPT.G_L2) + PARAMETERS_G:clone():mul(OPT.G_L2))
            end

            -- clamp G's Gradient to the range of -1.0 to +1.0
            if OPT.G_clamp ~= 0 then
                GRAD_PARAMETERS_G:clamp((-1)*OPT.G_clamp, OPT.G_clamp)
            end

            return f,GRAD_PARAMETERS_G
        end
        ------------------- end of eval functions ---------------------------

        ----------------------------------------------------------------------
        -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        -- Get half a minibatch of real, half fake
        for k=1, OPT.D_iterations do
            -- (1.1) Real data
            local inputIdx = 1
            local realDataSize = OPT.batchSize / 2
            for i = 1, realDataSize do
                local randomIdx = math.random(trainData:size())
                local trainingExample = trainData[randomIdx]
                --inputs[inputIdx] = trainingExample.uv:clone()
                inputs[inputIdx] = trainingExample.color:clone()
                condInputs[inputIdx] = trainingExample.grayscale:clone()
                targets[inputIdx] = Y_NOT_GENERATOR
                inputIdx = inputIdx + 1
            end

            -- (1.2) Sampled data
            noiseInputs:uniform(0, 1)
            for i = 1, realDataSize do
                local randomIdx = math.random(trainData:size())
                local trainingExample = trainData[randomIdx]
                condInputs[inputIdx] = trainingExample.grayscale:clone()
                inputIdx = inputIdx + 1
            end
            inputIdx = inputIdx - realDataSize

            local generatedUV = MODEL_G:forward({
                                    noiseInputs[{{realDataSize+1,2*realDataSize}}],
                                    condInputs[{{realDataSize+1,2*realDataSize}}]
                                })
            for i=1, realDataSize do
                inputs[inputIdx] = generatedUV[i]:clone()
                targets[inputIdx] = Y_GENERATOR
                inputIdx = inputIdx + 1
            end

            if OPT.D_optmethod == "sgd" then
                optim.sgd(fevalD, PARAMETERS_D, OPTSTATE.sgd.D)
            elseif OPT.D_optmethod == "adagrad" then
                optim.adagrad(fevalD, PARAMETERS_D, OPTSTATE.adagrad.D)
            elseif OPT.D_optmethod == "adadelta" then
                optim.adadelta(fevalD, PARAMETERS_D, OPTSTATE.adadelta.D)
            elseif OPT.D_optmethod == "adamax" then
                optim.adamax(fevalD, PARAMETERS_D, OPTSTATE.adamax.D)
            elseif OPT.D_optmethod == "adam" then
                optim.adam(fevalD, PARAMETERS_D, OPTSTATE.adam.D)
            elseif OPT.D_optmethod == "rmsprop" then
                optim.rmsprop(fevalD, PARAMETERS_D, OPTSTATE.rmsprop.D)
            else
                print("[Warning] Unknown optimizer method chosen for D.")
            end
        end

        ----------------------------------------------------------------------
        -- (2) Update G network: maximize log(D(G(z)))
        for k=1, OPT.G_iterations do
            noiseInputs:uniform(0, 1)
            targets:fill(Y_NOT_GENERATOR)
            for i=1,OPT.batchSize do
                local randomIdx = math.random(trainData:size())
                local trainingExample = trainData[randomIdx]
                condInputs[i] = trainingExample.grayscale:clone()
            end

            if OPT.G_optmethod == "sgd" then
                optim.sgd(fevalG_on_D, PARAMETERS_G, OPTSTATE.sgd.G)
            elseif OPT.G_optmethod == "adagrad" then
                optim.adagrad(fevalG_on_D, PARAMETERS_G, OPTSTATE.adagrad.G)
            elseif OPT.G_optmethod == "adadelta" then
                optim.adadelta(fevalG_on_D, PARAMETERS_G, OPTSTATE.adadelta.G)
            elseif OPT.G_optmethod == "adamax" then
                optim.adamax(fevalG_on_D, PARAMETERS_G, OPTSTATE.adamax.G)
            elseif OPT.G_optmethod == "adam" then
                optim.adam(fevalG_on_D, PARAMETERS_G, OPTSTATE.adam.G)
            elseif OPT.G_optmethod == "rmsprop" then
                optim.rmsprop(fevalG_on_D, PARAMETERS_G, OPTSTATE.rmsprop.G)
            else
                print("[Warning] Unknown optimizer method chosen for G.")
            end
        end

        batchIdx = batchIdx + 1
        -- display progress
        xlua.progress(batchIdx * OPT.batchSize, N_epoch * OPT.batchSize)
    end

    -- time taken
    time = sys.clock() - time
    if maxAccuracyD < 1.0 then
        print(string.format("<trainer> trained D %d of %d times.", countTrainedD, countTrainedD + countNotTrainedD))
    end

    -- print confusion matrix
    print("Confusion of D:")
    print(CONFUSION)
    local tV = CONFUSION.totalValid
    CONFUSION:zero()

    return tV
end

-- Show the activity of a network in windows (i.e. windows full of blinking dots).
-- The windows will automatically be reused.
-- Only the activity of the layer types nn.SpatialConvolution and nn.Linear will be shown.
-- Linear layers must have a minimum size to be shown (i.e. to not show the tiny output layers).
--
-- NOTE: This function can only visualize one network proberly while the program runs.
-- I.e. you can't call this function to show network A and then another time to show network B,
-- because the function tries to reuse windows and that will not work correctly in such a case.
--
-- NOTE: Old function, probably doesn't work anymore.
--
-- @param net The network to visualize.
-- @param minOutputs Minimum (output) size of a linear layer to be shown.
function adversarial.visualizeNetwork(net, minOutputs)
    if minOutputs == nil then
        minOutputs = 150
    end

    -- (Global) Table to save the window ids in, so that we can reuse them between calls.
    netvis_windows = netvis_windows or {}

    local modules = net:listModules()
    local winIdx = 1
    -- last module seems to have no output?
    for i=1,(#modules-1) do
        local t = torch.type(modules[i])
        local showTensor = nil
        -- This function only shows the activity of 2d convolutions and linear layers
        if t == 'nn.SpatialConvolution' then
            showTensor = modules[i].output[1]
        elseif t == 'nn.Linear' then
            local output = modules[i].output
            local shape = output:size()
            local nbValues = shape[2]

            if nbValues >= minOutputs and nbValues >= minOutputs then
                local nbRows = torch.floor(torch.sqrt(nbValues))
                while nbValues % nbRows ~= 0 and nbRows < nbValues do
                    nbRows = nbRows + 1
                end

                if nbRows >= nbValues then
                    showTensor = nil
                else
                    showTensor = output[1]:view(nbRows, nbValues / nbRows)
                end
            end
        end

        -- Show the layer outputs in a window
        -- Note that windows are reused if possible
        if showTensor ~= nil then
            netvis_windows[winIdx] = image.display{
                image=showTensor, zoom=1, nrow=32,
                min=-1, max=1,
                win=netvis_windows[winIdx], legend=t .. ' (#' .. i .. ')',
                padding=1
            }
            winIdx = winIdx + 1
        end
    end
end

return adversarial
