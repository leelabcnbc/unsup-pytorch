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
cmd:option('-nfiltersout', 4, 'number of output convolutional filters')
cmd:option('-kernelsize', 9, 'size of convolutional kernels')
cmd:option('-inputsize', 25, 'size of each input patch')
cmd:option('-lambda', 1, 'sparsity coefficient')
cmd:option('-datafile', 'tr-berkeley-N5K-M56x56-lcn.bin','Data set file')
cmd:option('-eta',0.01,'learning rate')
cmd:option('-momentum',0,'gradient momentum')
cmd:option('-decay',0,'weigth decay')
cmd:option('-maxiter',10,'max number of updates')
cmd:option('-statinterval',5000,'interval for saving stats and models')
cmd:option('-v', false, 'be verbose')
cmd:option('-wcar', '', 'additional flag to differentiate this run')
cmd:option('-conv', true, 'force convolutional dictionary')
cmd:text()

local params = cmd:parse(arg)

local rundir = cmd:string('unsup', params, {dir=true})
params.rundir = params.dir .. '/' .. rundir

os.execute('rm -rf ' .. params.rundir)

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
   -- need to be reversed (out/in instead of in/out).
   local conntable = nn.tables.full(params.nfiltersout, params.nfiltersin)
   mlp = unsup.SpatialConvFistaL1(conntable, params.kernelsize, params.kernelsize, params.inputsize, params.inputsize, params.lambda, {verbose=true})
   -- fuck. default weights don't play well.
   mlp:normalize()
   -- fuck. default bias is not zero.
   mlp.D.bias:zero()
   -- convert to conv format.
   data:conv()
end

-- do learrning rate hacks
nnhacks()

function train(module,dataset)

   local avTrainingError = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
   local avFistaIterations = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
   local currentLearningRate = params.eta

   local function updateSample(input, target, eta)
     -- hack for second arg. 17=25-9+1
     local err,h = module:updateOutput(input, torch.zeros(4,17,17))
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
     table.insert(all_data, example[1]:view(1,1,25,25):clone())
     
     -- get weight before --
     table.insert(all_weight, module.D.weight:view(1,4,9,9):clone())
     local serr, siter = updateSample(example[1], example[2] ,currentLearningRate)
     -- get grad_weight later --
     table.insert(all_grad_weight, module.D.gradWeight:view(1,4,9,9):clone())
     -- also, get serr to check L1 stuff.
     table.insert(all_serr, serr)
     err = err + serr
     iter = iter + siter
        
        
     if math.fmod(t , params.statinterval) == 0 then
       avTrainingError[t/params.statinterval] = err/params.statinterval
       avFistaIterations[t/params.statinterval] = iter/params.statinterval
       -- report
       print('# iter=' .. t .. ' eta = ' .. currentLearningRate .. ' current error = ' .. err)
       err = 0
       iter = 0
     end
  end
    npy4th.savenpy('demo_fista_conv_debug_lam1_data.npy', torch.cat(all_data,1))
    npy4th.savenpy('demo_fista_conv_debug_lam1_weight.npy', torch.cat(all_weight,1))
    npy4th.savenpy('demo_fista_conv_debug_lam1_grad_weight.npy', torch.cat(all_grad_weight,1))
    npy4th.savenpy('demo_fista_conv_debug_lam1_serr.npy', torch.FloatTensor(all_serr))
end

train(mlp,data)
