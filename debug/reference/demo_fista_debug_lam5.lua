require 'unsup'
require 'image'
require 'gnuplot'
npy4th = require 'npy4th'

dofile 'demo_data.lua'
dofile 'demo_utils.lua'
if not arg then arg = {} end

cmd = torch.CmdLine()

cmd:text()
cmd:text()
cmd:text('Training a simple sparse coding dictionary on Berkeley images')
cmd:text()
cmd:text()
cmd:text('Options')
cmd:option('-dir','outputs', 'subdirectory to save experimens in')
cmd:option('-seed', 123211, 'initial random seed')
cmd:option('-nfiltersin', 1, 'number of input convolutional filters')
cmd:option('-nfiltersout', 32, 'number of output convolutional filters')
cmd:option('-kernelsize', 9, 'size of convolutional kernels')
cmd:option('-inputsize', 9, 'size of each input patch')
-- intially I used 5, but that gives trivial results ... all being zero
cmd:option('-lambda', 0.5, 'sparsity coefficient')
cmd:option('-datafile', 'tr-berkeley-N5K-M56x56-lcn.bin','Data set file')
cmd:option('-eta',0.01,'learning rate')
cmd:option('-momentum',0,'gradient momentum')
cmd:option('-decay',0,'weigth decay')
cmd:option('-maxiter',10,'max number of updates')
cmd:option('-statinterval',5000,'interval for saving stats and models')
cmd:option('-v', false, 'be verbose')
cmd:option('-wcar', '', 'additional flag to differentiate this run')
cmd:option('-conv', false, 'force convolutional dictionary')
cmd:text()

local params = cmd:parse(arg)

local rundir = cmd:string('unsup', params, {dir=true})
params.rundir = params.dir .. '/' .. rundir

if paths.dirp(params.rundir) then
   error('This experiment is already done!!!')
end

os.execute('mkdir -p ' .. params.rundir)
cmd:log(params.rundir .. '/log', params)

-- init random number generator
torch.manualSeed(params.seed)

-- create the dataset
data = getdata(params.datafile, params.inputsize)

-- creat unsup stuff
if params.inputsize == params.kernelsize and params.conv == false then
   print('Linear sparse coding')
   mlp = unsup.LinearFistaL1(params.inputsize*params.inputsize, params.nfiltersout, params.lambda,
    {verbose=true})
else
   print('Convolutional sparse coding')
   mlp = unsup.SpatialConvFistaL1(params.nfiltersin, params.nfiltersout, params.kernelsize, params.kernelsize, params.inputsize, params.inputsize, params.lambda)
end

-- do learrning rate hacks
nnhacks()

function train(module,dataset)

   local avTrainingError = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
   local avFistaIterations = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
   local currentLearningRate = params.eta

   local function updateSample(input, target, eta)
     local err,h = module:updateOutput(input, target)
     module:zeroGradParameters()
     module:updateGradInput(input, target)
     module:accGradParameters(input, target)
     module:updateParameters(eta)
     return err, #h
  end

  local err = 0
  local iter = 0
  local all_data = {}
  local all_weight = {}
  local all_grad_weight = {}
  local all_serr = {}
  for t = 1,params.maxiter do

     local example = dataset[t]
     table.insert(all_data, example[1]:view(1,81):clone())
     
     -- get weight before --
     table.insert(all_weight, module.D.weight:view(1,81,32):clone())
     local serr, siter = updateSample(example[1], example[2] ,currentLearningRate)
     -- get grad_weight later --
     table.insert(all_grad_weight, module.D.gradWeight:view(1,81,32):clone())
     -- also, get serr to check L1 stuff.
     table.insert(all_serr, serr)
     err = err + serr
     iter = iter + siter
  end
    npy4th.savenpy('demo_fista_debug_lam5_data.npy', torch.cat(all_data,1))
    npy4th.savenpy('demo_fista_debug_lam5_weight.npy', torch.cat(all_weight,1))
    npy4th.savenpy('demo_fista_debug_lam5_grad_weight.npy', torch.cat(all_grad_weight,1))
    npy4th.savenpy('demo_fista_debug_lam5_serr.npy', torch.FloatTensor(all_serr))
end

train(mlp,data)
