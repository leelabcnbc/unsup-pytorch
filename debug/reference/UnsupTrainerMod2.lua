npy4th = require 'npy4th'
local UnsupTrainer = torch.class('unsup.UnsupTrainerMod2')

function UnsupTrainer:__init(module,data)

   local x,dx,ddx = module:getParameters()
   self.parameters = {x,dx,ddx}
   if not self.parameters or #self.parameters == 0 then
      error(' I could not get parameters from module...')
   end
   self.module = module
   self.data = data
end

function UnsupTrainer:train(params)
   -- essential stuff
   local data = self.data
   local eta = params.eta
   local etadecay = params.etadecay or 0
   local maxiter = params.maxiter
   local statinterval = params.statinterval or math.ceil(maxiter/100)
   local etadecayinterval = params.etadecayinterval or statinterval
   -- optional hessian stuff
   local dohessian = params.hessian or false
   local hessianinterval = params.hessianinterval or statinterval

   if not dohessian then self.parameters[3] = nil end

   local age = 1
   local err = 0
    
    
   local all_data = {}
   local all_weight = {}
   local all_grad_weight = {}
   local all_serr = {}
   
   local all_encoder_weight = {}
   local all_encoder_bias = {}
   local all_encoder_scale = {}
    
    
   while age <= maxiter do
      
      -- DATA
      local ex = data[age]
        
        
      table.insert(all_data, ex[1]:view(1,1,25,25):clone())
     
     -- get weight before --
      table.insert(all_weight, self.module.decoder.D.weight:view(1,4,9,9):clone())
      table.insert(all_encoder_weight, self.module.encoder.modules[1].weight:view(1,4,9,9):clone())
      table.insert(all_encoder_bias, self.module.encoder.modules[1].bias:view(1,4):clone())
      table.insert(all_encoder_scale, self.module.encoder.modules[3].weight:view(1,4):clone()) 

      -- SGD UPDATE
      local sres = self:trainSample(ex,eta)
      local serr = sres[1]
      err = err + serr
      
        
      -- get grad_weight later --
      table.insert(all_grad_weight, self.module.decoder.D.gradWeight:view(1,4,9,9):clone())
      -- also, get serr to check L1 stuff.
      table.insert(all_serr, serr)


      age = age + 1
   end
   -- save all
   npy4th.savenpy('demo_psd_conv_debug_beta5_data.npy', torch.cat(all_data,1))
   npy4th.savenpy('demo_psd_conv_debug_beta5_weight.npy', torch.cat(all_weight,1))
    
   npy4th.savenpy('demo_psd_conv_debug_beta5_encoder_weight.npy', torch.cat(all_encoder_weight,1))
   npy4th.savenpy('demo_psd_conv_debug_beta5_encoder_bias.npy', torch.cat(all_encoder_bias,1))
   npy4th.savenpy('demo_psd_conv_debug_beta5_encoder_scale.npy', torch.cat(all_encoder_scale,1))
    
   npy4th.savenpy('demo_psd_conv_debug_beta5_grad_weight.npy', torch.cat(all_grad_weight,1))
   npy4th.savenpy('demo_psd_conv_debug_beta5_serr.npy', torch.FloatTensor(all_serr))
end


function UnsupTrainer:trainSample(ex, eta)
   local module = self.module
   local parameters = self.parameters

   local input = ex[1]
   local target = ex[2]

   local x = parameters[1]
   local dx = parameters[2]
   local ddx = parameters[3]

   local res = {module:updateOutput(input, target)}
   -- clear derivatives
   dx:zero()
   module:updateGradInput(input, target)
   module:accGradParameters(input, target)

   if dx:max() > 100 or dx:min() < -100 then
      print('oops large dx ' .. dx:max() .. ' ' .. dx:min())
   end

   if torch.ne(dx,dx):sum()  > 0 then
      print('oops nan dx')
      --torch.save('error.bin',module)
      error('oops nan dx')
   end

   --print('k min/max (before) =',module.decoder.D.weight:min(),module.decoder.D.weight:max())
   -- do update
   if not ddx then
      -- regular sgd
      x:add(-eta,dx)
   else
      -- diag hessian
      x:addcdiv(-eta,dx,ddx)
   end
   if torch.ne(x,x):sum()  > 0 then
      print('oops nan x')
      --torch.save('error.bin',module)
      error('oops nan x')
   end
   module:normalize()
   -- print('k min/max (after) =',module.decoder.D.weight:min(),module.decoder.D.weight:max())
   -- print('k norm=',module.decoder.D.weight[1]:norm())
   -- print('code min/max (after) =',module.decoder.code:min(),module.decoder.code:max())
   if torch.ne(x,x):sum()  > 0 then
      print('oops nan x norm')
      --torch.save('error.bin',module)
      error('oops nan x norm')
   end
   return res
end
