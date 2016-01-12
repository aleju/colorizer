require 'torch'
require 'image'
require 'pl' -- this is somehow responsible for lapp working in qlua mode
require 'paths'
ok, DISP = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
ADVERSARIAL = require 'adversarial_rgb'
DATASET = require 'dataset_rgb'
NN_UTILS = require 'utils.nn_utils'
MODELS = require 'models_rgb'


----------------------------------------------------------------------
-- parse command-line options
OPT = lapp[[
  --save             (default "logs")       subdirectory to save logs
  --saveFreq         (default 30)           save every saveFreq epochs
  --network          (default "")           reload pretrained network
  --G_pretrained_dir (default "logs")
  --noplot                                  plot while training
  --D_sgd_lr         (default 0.02)         D SGD learning rate
  --G_sgd_lr         (default 0.02)         G SGD learning rate
  --D_sgd_momentum   (default 0)            D SGD momentum
  --G_sgd_momentum   (default 0)            G SGD momentum
  --batchSize        (default 32)           batch size
  --N_epoch          (default 30)           Number of batches per epoch
  --G_L1             (default 0)            L1 penalty on the weights of G
  --G_L2             (default 0e-6)         L2 penalty on the weights of G
  --D_L1             (default 0e-7)         L1 penalty on the weights of D
  --D_L2             (default 1e-4)         L2 penalty on the weights of D
  --D_iterations     (default 1)            number of iterations to optimize D for
  --G_iterations     (default 1)            number of iterations to optimize G for
  --D_maxAcc         (default 1.01)         Deactivate learning of D while above this threshold
  --D_clamp          (default 1)            Clamp threshold for D's gradient (+/- N)
  --G_clamp          (default 5)            Clamp threshold for G's gradient (+/- N)
  --D_optmethod      (default "adam")       sgd|adagrad|adadelta|adamax|adam|rmsprob
  --G_optmethod      (default "adam")       sgd|adagrad|adadelta|adamax|adam|rmsprob
  --threads          (default 4)            number of threads
  --gpu              (default 0)            gpu to run on (default cpu)
  --noiseDim         (default 100)          dimensionality of noise vector
  --window           (default 3)            window id of sample image
  --seed             (default 1)            seed for the RNG
  --nopretraining                           Whether to deactivate loading of pretrained networks
  --height           (default 64)
  --width            (default 64)
]]

NORMALIZE = false
START_TIME = os.time()

if OPT.gpu < 0 or OPT.gpu > 3 then OPT.gpu = false end
print(OPT)

-- fix seed
math.randomseed(OPT.seed)
torch.manualSeed(OPT.seed)

-- threads
torch.setnumthreads(OPT.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- possible output of disciminator
CLASSES = {"0", "1"}
Y_GENERATOR = 0
Y_NOT_GENERATOR = 1

-- axis of images: 3 channels, <scale> height, <scale> width
IMG_DIMENSIONS = {3, OPT.height, OPT.width}
COND_DIM = {1, OPT.height, OPT.width}
NOISE_DIM = {1, OPT.height, OPT.width}

-- size in values/pixels per input image (channels*height*width)
--INPUT_SZ = IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2] * IMG_DIMENSIONS[3]

----------------------------------------------------------------------
-- get/create dataset
----------------------------------------------------------------------
DATASET.setFileExtension("jpg")
DATASET.setHeight(IMG_DIMENSIONS[2])
DATASET.setWidth(IMG_DIMENSIONS[3])
DATASET.setDirs({"/media/aj/ssd2a/nlp/python/git/face-generator/dataset/out_aug_64x64"})
--DATASET.setDirs({"/media/aj/ssd2a/nlp/python/git/christmas-generator/dataset/preprocessed/christmas-trees"})
--DATASET.setDirs({"/media/aj/ssd2a/nlp/python/git/christmas-generator/dataset/preprocessed/baubles"})
--DATASET.setDirs({"/media/aj/ssd2a/nlp/python/git/sky-generator/dataset/out_aug_32x64"})
--DATASET.setDirs({"/media/aj/ssd2a/nlp/python/git/cat-generator/dataset/out_aug_64x64"})
--[[
DATASET.setDirs({
    "/media/aj/grab/ml/datasets/10k_cats/CAT_00",
    "/media/aj/grab/ml/datasets/10k_cats/CAT_01",
    "/media/aj/grab/ml/datasets/10k_cats/CAT_02",
    "/media/aj/grab/ml/datasets/10k_cats/CAT_03",
    "/media/aj/grab/ml/datasets/10k_cats/CAT_04",
    "/media/aj/grab/ml/datasets/10k_cats/CAT_05",
    "/media/aj/grab/ml/datasets/10k_cats/CAT_06"
})
--]]
----------------------------------------------------------------------

-- run on gpu if chosen
-- We have to load all kinds of libraries here, otherwise we risk crashes when loading
-- saved networks afterwards
print("<trainer> starting gpu support...")
require 'nn'
require 'cutorch'
require 'cunn'
require 'dpnn'
if OPT.gpu then
    cutorch.setDevice(OPT.gpu + 1)
    cutorch.manualSeed(OPT.seed)
    print(string.format("<trainer> using gpu device %d", OPT.gpu))
end
torch.setdefaulttensortype('torch.FloatTensor')

function main()
    ----------------------------------------------------------------------
    -- Load / Define network
    ----------------------------------------------------------------------

    -- load previous networks (D and G)
    -- or initialize them new
    if OPT.network ~= "" then
        print(string.format("<trainer> reloading previously trained network: %s", OPT.network))
        local tmp = torch.load(OPT.network)
        MODEL_D = tmp.D
        MODEL_G = tmp.G
        EPOCH = tmp.epoch + 1
        VIS_NOISE_INPUTS = tmp.vis_noise_inputs
        if NORMALIZE then
            NORMALIZE_MEAN = tmp.normalize_mean
            NORMALIZE_STD = tmp.normalize_std
        end

        if OPT.gpu == false then
            MODEL_D:float()
            MODEL_G:float()
        end
    else
        local pt_filename = paths.concat(OPT.save, string.format('pretrained_%dx%dx%d_nd%d.net', IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3], OPT.noiseDim))
        -- pretrained via pretrain_with_previous_net.lua ?
        if not OPT.nopretraining and paths.filep(pt_filename) then
            local tmp = torch.load(pt_filename)
            MODEL_D = tmp.D
            MODEL_G = tmp.G
            MODEL_D:training()
            MODEL_G:training()
            if OPT.gpu == false then
                MODEL_D:float()
                MODEL_G:float()
            end
        else
            --------------
            -- D
            --------------
            MODEL_D = MODELS.create_D(IMG_DIMENSIONS, OPT.gpu ~= false)

            --------------
            -- G
            --------------
            local g_pt_filename = paths.concat(OPT.G_pretrained_dir, string.format('g_pretrained_%dx%dx%d_nd%d.net', IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3], OPT.noiseDim))
            if not OPT.nopretraining and paths.filep(g_pt_filename) then
                -- Load a pretrained version of G
                print("<trainer> loading pretrained G...")
                local tmp = torch.load(g_pt_filename)
                MODEL_G = tmp.G
                MODEL_G:training()
                if OPT.gpu == false then
                    MODEL_G:float()
                end
            else
                print("<trainer> Note: Did not find pretrained G")
                MODEL_G = MODELS.create_G(IMG_DIMENSIONS, OPT.gpu ~= false)
            end
        end
    end

    print(MODEL_G)
    print(MODEL_D)

    -- count free parameters in D/G
    print(string.format('Number of free parameters in D: %d', NN_UTILS.getNumberOfParameters(MODEL_D)))
    print(string.format('Number of free parameters in G: %d', NN_UTILS.getNumberOfParameters(MODEL_G)))

    -- loss function: negative log-likelihood
    CRITERION = nn.BCECriterion()

    -- retrieve parameters and gradients
    PARAMETERS_D, GRAD_PARAMETERS_D = MODEL_D:getParameters()
    PARAMETERS_G, GRAD_PARAMETERS_G = MODEL_G:getParameters()

    -- this matrix records the current confusion across classes
    CONFUSION = optim.ConfusionMatrix(CLASSES)

    -- Set optimizer states
    OPTSTATE = {
        adagrad = { D = {}, G = {} },
        adadelta = { D = {}, G = {} },
        adamax = { D = {}, G = {} },
        adam = { D = {}, G = {} },
        rmsprop = {D = {}, G = {}},
        sgd = {
            D = {learningRate = OPT.D_sgd_lr, momentum = OPT.D_sgd_momentum},
            G = {learningRate = OPT.G_sgd_lr, momentum = OPT.G_sgd_momentum}
        }
    }

    if NORMALIZE then
        if NORMALIZE_MEAN == nil then
            TRAIN_DATA = DATASET.loadRandomImages(10000)
            NORMALIZE_MEAN, NORMALIZE_STD = TRAIN_DATA.normalize()
        end
    end

    if EPOCH == nil then
        EPOCH = 1
    end

    PLOT_DATA = {}
    if VIS_NOISE_INPUTS == nil then
        VIS_NOISE_INPUTS = NN_UTILS.createNoiseInputs(100)
    end

    EXAMPLE_IMAGES = DATASET.loadRandomImages(48)

    -- training loop
    while true do
        print('Loading new training data...')
        TRAIN_DATA = DATASET.loadRandomImages(OPT.N_epoch * OPT.batchSize)
        if NORMALIZE then
            TRAIN_DATA.normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        end

        -- Show images and plots if requested
        if not OPT.noplot then
            --visualizeProgress(MODEL_G, MODEL_D, VIS_NOISE_INPUTS, TRAIN_DATA)
            visualizeProgressConditional()
        end

        -- Train D and G
        -- ... but train D only while having an accuracy below OPT.D_maxAcc
        --     over the last math.max(20, math.min(1000/OPT.batchSize, 250)) batches
        ADVERSARIAL.train(TRAIN_DATA, OPT.D_maxAcc, math.max(20, math.min(1000/OPT.batchSize, 250)))

        -- Save current net
        if EPOCH % OPT.saveFreq == 0 then
            local filename = paths.concat(OPT.save, 'adversarial.net')
            saveAs(filename)
        end

        EPOCH = EPOCH + 1
    end
end

-- Save the current models G and D to a file.
-- @param filename The path to the file
function saveAs(filename)
    os.execute(string.format("mkdir -p %s", sys.dirname(filename)))
    if paths.filep(filename) then
      os.execute(string.format("mv %s %s.old", filename, filename))
    end
    print(string.format("<trainer> saving network to %s", filename))
    NN_UTILS.prepareNetworkForSave(MODEL_G)
    NN_UTILS.prepareNetworkForSave(MODEL_D)
    torch.save(filename, {D = MODEL_D, G = MODEL_G, opt = OPT, plot_data = PLOT_DATA, epoch = EPOCH, vis_noise_inputs = VIS_NOISE_INPUTS, normalize_mean=NORMALIZE_MEAN, normalize_std=NORMALIZE_STD})
end

-- Visualizes the current training status via Display (based on gfx.js) in the browser.
-- It shows:
--   Images generated from random noise (the noise vectors are set once at the start of the
--   training, so the images should end up similar at each epoch)
--   Images that were deemed "good" by D
--   Images that were deemed "bad" by D
--   Original images from the training set (as comparison)
-- @param noiseInputs The noise vectors for the random images.
-- @returns void
function visualizeProgress(modelG, modelD, noiseInputs, trainData)
    -- deactivate dropout
    modelG:evaluate()
    modelD:evaluate()

    -- Generate a synthetic test image as sanity test
    -- This should be deemed very bad by D
    local sanityTestImage = torch.Tensor(IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
    sanityTestImage:uniform(0.0, 0.50)
    for i=1,IMG_DIMENSIONS[2] do
        for j=1,IMG_DIMENSIONS[3] do
            if i == j then
                sanityTestImage[1][i][j] = 1.0
            elseif i % 4 == 0 and j % 4 == 0 then
                sanityTestImage[1][i][j] = 0.5
            end
        end
    end

    -- Collect original example images from the training set
    local trainImages = trainData[{{1, 50}, {}, {}, {}}]:clone()

    -- Generate images from G based on the provided noiseInputs
    local rndImages = NN_UTILS.createImagesFromNoise(noiseInputs)

    -- Place the sanity test image and one original image from the training corpus among
    -- the random Images. The first should be deemed bad by D, the latter as good.
    -- Then find good and bad images (according to D) among the randomly generated ones
    -- Note: has to happen before toRgb() as that would change the color space of the images
    local rndImagesClone = rndImages:clone()
    rndImagesClone[rndImagesClone:size(1)-1] = trainImages[1] -- one real face as sanity test
    rndImagesClone[rndImagesClone:size(1)] = sanityTestImage -- synthetic non-face as sanity test
    local goodImages, _ = NN_UTILS.sortImagesByPrediction(rndImagesClone, false, 50)
    local badImages, _ = NN_UTILS.sortImagesByPrediction(rndImagesClone, true, 50)

    if rndImages:ne(rndImages):sum() > 0 then
        print(string.format("[visualizeProgress] Generated images contain NaNs"))
    end

    DISP.image(NN_UTILS.toRgb(rndImages, OPT.colorSpace), {win=OPT.window+1, width=IMG_DIMENSIONS[3]*15, title="Generated images (epoch " .. EPOCH .. ")"})
    DISP.image(NN_UTILS.toRgb(goodImages, OPT.colorSpace), {win=OPT.window+2, width=IMG_DIMENSIONS[3]*15, title="Best samples (first is best) (epoch " .. EPOCH .. ")"})
    DISP.image(NN_UTILS.toRgb(badImages, OPT.colorSpace), {win=OPT.window+3, width=IMG_DIMENSIONS[3]*15, title="Worst samples (first is worst) (epoch " .. EPOCH .. ")"})
    DISP.image(NN_UTILS.toRgb(trainImages, OPT.colorSpace), {win=OPT.window+4, width=IMG_DIMENSIONS[3]*15, title="original images from training set"})

    NN_UTILS.saveImagesAsGrid(string.format("%s/images/%d_%05d.png", OPT.save, START_TIME, EPOCH), NN_UTILS.toRgb(rndImages, OPT.colorSpace), 10, 10, EPOCH)
    NN_UTILS.saveImagesAsGrid(string.format("%s/images_good/%d_%05d.png", OPT.save, START_TIME, EPOCH), NN_UTILS.toRgb(goodImages, OPT.colorSpace), 7, 7, EPOCH)
    NN_UTILS.saveImagesAsGrid(string.format("%s/images_bad/%d_%05d.png", OPT.save, START_TIME, EPOCH), NN_UTILS.toRgb(badImages, OPT.colorSpace), 7, 7, EPOCH)

    -- reactivate dropout
    modelG:training()
    modelD:training()
end

-- Get examples to plot.
-- Returns a list of the pattern
--  [i] Coarse/blurry image,
--  [i+1] Fine image (Coarse + true diff)
--  [i+2] Coarse + diff generated by G
--  [i+3] True diff
--  [i+4] Diff generated by G
-- @param ds Dataset as list of examples that have attributes .coarse .fine and .diff
-- @param N Number of samples to prepare
function getSamples(ds, N)
    --local N = N or 8
    local N = EXAMPLE_IMAGES:size()
    local ds = EXAMPLE_IMAGES
    local noiseInputs = torch.Tensor(N, NOISE_DIM[1], NOISE_DIM[2], NOISE_DIM[3])
    local condInputs = torch.Tensor(N, COND_DIM[1], COND_DIM[2], COND_DIM[3])
    --local gt_diff = torch.Tensor(N, 2, IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
    local gt = torch.Tensor(N, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])

    -- Generate samples
    noiseInputs:uniform(0, 1)
    for i=1,N do
        --local idx = math.random(ds:size())
        local idx = i
        local example = ds[idx]
        condInputs[i] = example.grayscale:clone()
        gt[i] = example.color:clone()
        --gt_diff[i] = example.uv:clone()
    end
    local samples = MODEL_G:forward({noiseInputs, condInputs})

    local to_plot = {}
    for i=1,N do
        --local withColor = torch.cat(condInputs[i]:float(), samples[i]:float(), 1)
        local withColor = samples[i]:clone()
        to_plot[#to_plot+1] = NN_UTILS.switchColorSpaceSingle(condInputs[i]:float(), "y", "rgb")
        to_plot[#to_plot+1] = gt[i]:float()
        to_plot[#to_plot+1] = withColor
        --print(to_plot[#to_plot-2]:size(), to_plot[#to_plot-1]:size(), to_plot[#to_plot-0]:size())
        --to_plot[#to_plot+1] = gt_diff[i]:float()
        --to_plot[#to_plot+1] = samples[i]:float()
    end
    return to_plot
end

function visualizeProgressConditional()
    -- Show images and their refinements for the validation and training set
    --local toPlotVal = getSamples(VAL_DATA, 20)
    local toPlotTrain = getSamples()
    --DISP.image(toPlotVal, {win=OPT.window, width=2*10*IMG_DIMENSIONS[3], title=string.format("[VAL] Coarse, GT, G img, GT diff, G diff (%s epoch %d)", OPT.save, EPOCH)})
    DISP.image(toPlotTrain, {win=OPT.window+1, width=14*IMG_DIMENSIONS[3], title=string.format("[TRAIN] original grayscale, original color, auto-colorized (%s epoch %d)", OPT.save, EPOCH)})
end

main()
