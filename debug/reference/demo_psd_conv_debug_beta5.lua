require 'unsup'
require 'image'
require 'gnuplot'

dofile 'demo_data.lua'
dofile 'UnsupTrainerMod2.lua'

if not arg then arg = {} end

cmd = torch.CmdLine()

cmd:text()
cmd:text()
cmd:text('Training a PSD Model')
cmd:text()
cmd:text()
cmd:text('Options')
cmd:option('-dir','outputs.psd', 'subdirectory to save experimens in')
cmd:option('-data','tr-berkeley-N5K-M56x56-lcn.bin','Data set file')
cmd:option('-seed', 123211, 'initial random seed')
cmd:option('-inputsize',25, 'size of input patches')
cmd:option('-nfiltersin', 1, 'number of input convolutional filters')
cmd:option('-nfiltersout', 4, 'number of output convolutional filters')
cmd:option('-conntable','','connection table for the unsupervised module')
cmd:option('-kernelsize', 9, 'size of convolutional kernels')
cmd:option('-lambda', 1, 'sparsity coefficient')
cmd:option('-beta', 5, 'prediction error coefficient')
cmd:option('-encoderType','tanh','Encoder Architecture')
cmd:option('-eta',0.002,'learning rate')
cmd:option('-etadecay',0,'learning rate decay')
cmd:option('-etadecayinterval',10000,'learning rate decay interval')
cmd:option('-maxiter',10,'max number of updates')
cmd:option('-statinterval',5000,'interval for saving stats and models')
cmd:option('-v', false, 'be verbose')
cmd:option('-wcar', '', 'additional flag to differentiate this run')
cmd:option('-openmp',false,'Use OpenMP')
cmd:option('-nThread',1,'Number of threads for openmp')
cmd:option('-hessian',false,'Compute Diagonal Hessian Approximation')
cmd:option('-hessianinterval',10000,'Compute Diagonal Hessian Approximation at every this many samples')
cmd:option('-minhessian',0.02,'Min hessian to avoid extreme speed up')
cmd:option('-maxhessian',500,'Max hessian to avoid extreme slow down')
cmd:option('-linear',false,'Train a linear model')
cmd:text()

local params = cmd:parse(arg)

-- if not params.hessian then
--    error('convolutional psd runs much much much faster with psd')
-- end
nn.hessian.enable()

if params.openmp or params.nThread > 1 then
   torch.setDefaultNumThreads(params.nThread)
   print('Using OpenMP')
end

if params.linear then
   params.inputsize = params.kernelsize
end
if not paths.filep(params.data) then
   print('Datafile does not exist : ' .. params.data)
   print('You can get sample datafile from http://cs.nyu.edu/~koray/publis/code/tr-berkeley-N5K-M56x56-lcn.bin')
   error('Aborting: no data')
end

local rundir = cmd:string('psd', params, {dir=true,lambda=false,encoderType=false,kernelsize=false})
params.rundir = paths.concat(params.dir,rundir)

if paths.dirp(params.rundir) then
   error('This experiment is already done!!!')
end

os.execute('mkdir -p ' .. params.rundir)
cmd:addTime('psd')
cmd:log(paths.concat(params.rundir, 'log'), params)

-- init random number generator
torch.manualSeed(params.seed)

-- create the dataset
data = getdata(params.data,params.inputsize)--imutils.mapdata(params.data)
data:conv()

local ex = data[1][1]
local exwidth = ex:size(ex:dim())
local exheight = ex:size(ex:dim()-1)

-- creat unsup stuff
psdparam = {}
psdparam.encoderType = params.encoderType
psdparam.verbose = params.v
print('Convolutional psd')
local conntable
if params.conntable ~= '' and paths.filep(params.conntable) then
   print('Using connection table')
   conntable = torch.load(params.conntable)
else
   conntable = nn.tables.full(params.nfiltersin, params.nfiltersout)
end
mlp = unsup.ConvPSD(conntable, params.kernelsize, params.kernelsize, exwidth, exheight, params.lambda, params.beta, psdparam)
mlp.decoder.D.bias:zero()
mlp:initDiagHessianParameters()

trainer = unsup.UnsupTrainerMod2(mlp,data)
----------------------------------------------------------------
-- LOGGING STUFF
----------------------------------------------------------------
local nerr = math.ceil(params.maxiter/params.statinterval)
local allerr = torch.Tensor(6,nerr):zero()
logs = {}
logs.toterr         = allerr[1]
logs.prediction     = allerr[2]
logs.reconstruction = allerr[3]
logs.sparsity       = allerr[4]
logs.niter          = allerr[5]
logs.nline          = allerr[6]

if params.hessianinterval == 0 then params.hessianinterval = nil end

print('Starting Training')
function train()
   os.execute('mkdir -p ' .. paths.concat(params.rundir,'source'))
   os.execute('cp *.lua ' .. paths.concat(params.rundir,'source/.'))
   trainer:train{eta=params.eta,
		 etadecay = params.etadecay,
		 etadecayinterval = params.etadecayinterval,
		 maxiter=params.maxiter,
		 statinterval=params.statinterval,
		 hessian=params.hessian,
		 hessianinterval=params.hessianinterval,
		 minhessian = params.minhessian,
		 maxhessian = params.maxhessian}
end

local res,err=pcall(train)

if not res then
   print('Training failed')
   print(err)
else
   print('Training done')
end
print('Saving mlp')
torch.save(paths.concat(params.rundir,'model.bin'),mlp)
torch.save(paths.concat(params.rundir,'tr.bin'),trainer)

