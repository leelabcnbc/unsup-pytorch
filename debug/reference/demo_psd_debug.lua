require 'unsup'
require 'image'
require 'gnuplot'
npy4th = require 'npy4th'

dofile 'demo_data.lua'

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
cmd:option('-lambda', 1, 'sparsity coefficient')
cmd:option('-beta', 1, 'prediction error coefficient')
cmd:option('-datafile', 'tr-berkeley-N5K-M56x56-lcn.bin','Data set file')
cmd:option('-eta',0.01,'learning rate')
cmd:option('-eta_encoder',0,'encoder learning rate')
cmd:option('-momentum',0,'gradient momentum')
cmd:option('-decay',0,'weigth decay')
cmd:option('-maxiter',10,'max number of updates')
cmd:option('-statinterval',5000,'interval for saving stats and models')
cmd:option('-v', false, 'be verbose')
cmd:option('-wcar', '', 'additional flag to differentiate this run')
cmd:option('-conv', false, 'force convolutional dictionary')
cmd:text()

local params = cmd:parse(arg)

local rundir = cmd:string('psd', params, {dir=true})
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
   print('Linear psd')
   mlp = unsup.LinearPSD(params.inputsize*params.inputsize, params.nfiltersout, params.lambda, params.beta )
else
   print('Convolutional psd')
   local conntable = nn.tables.full(params.nfiltersin, params.nfiltersout)
   mlp = unsup.ConvPSD(conntable, params.kernelsize, params.kernelsize, params.inputsize, params.inputsize, params.lambda, params.beta)
   -- convert dataset to convolutional
   data:conv()
end

-- learning rates
if params.eta_encoder == 0 then params.eta_encoder = params.eta end
params.eta = torch.Tensor({params.eta_encoder, params.eta})

-- do learrning rate hacks
-- kex.nnhacks()

function train(module,dataset)

   local avTrainingError = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
   local avFistaIterations = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
   local currentLearningRate = params.eta
   
   local function updateSample(input, target, eta)
      print('updateOutput')
      local err,h = module:updateOutput(input, target)
      module:zeroGradParameters()
      print('updateGradInput')
      module:updateGradInput(input, target)
      print('accGradParameters')
      module:accGradParameters(input, target)
      print('updateParameters')
      module:updateParameters(eta)
      return err, #h
   end

   local err = 0
   local iter = 0
   local all_data = {}
   local all_weight = {}
   
   local all_encoder_weight = {}
   local all_encoder_bias = {}
   local all_encoder_scale = {}
   
   local all_grad_weight = {}
   local all_serr = {}
    
   for t = 1,params.maxiter do

      local example = dataset[t]
        
      table.insert(all_data, example[1]:view(1,81):clone())
     
     -- get weight before --
      table.insert(all_weight, module.decoder.D.weight:view(1,81,32):clone())
      table.insert(all_encoder_weight, module.encoder.weight:view(1,32,81):clone())
      table.insert(all_encoder_bias, module.encoder.bias:view(1,32):clone())
--       table.insert(all_encoder_scale, module.encoder.modules[3].weight:view(1,32):clone()) 
        
      local serr, siter = updateSample(example[1], example[2] ,currentLearningRate)
      -- get grad_weight later --
      table.insert(all_grad_weight, module.decoder.D.gradWeight:view(1,81,32):clone())
      -- also, get serr to check L1 stuff.
      table.insert(all_serr, serr)
      err = err + serr
      iter = iter + siter

   end
    npy4th.savenpy('demo_psd_debug_data.npy', torch.cat(all_data,1))
    npy4th.savenpy('demo_psd_debug_weight.npy', torch.cat(all_weight,1))
    
    npy4th.savenpy('demo_psd_debug_encoder_weight.npy', torch.cat(all_encoder_weight,1))
    npy4th.savenpy('demo_psd_debug_encoder_bias.npy', torch.cat(all_encoder_bias,1))
--     npy4th.savenpy('demo_psd_debug_encoder_scale.npy', torch.cat(all_encoder_scale,1))
    
    npy4th.savenpy('demo_psd_debug_grad_weight.npy', torch.cat(all_grad_weight,1))
    npy4th.savenpy('demo_psd_debug_serr.npy', torch.FloatTensor(all_serr))
end

train(mlp,data)
