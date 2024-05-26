import torch
import torch.nn as nn
import numpy as np
import math
import librosa

class LearnOVSTFT2MELChannel(nn.Module):
    def __init__(self, nfreq, nfilters, n_channels, dense_layer_width, \
        dense_layer_width2, use_res=False, sample_rate=16000):
        super().__init__()
        self.nfreq = nfreq
        self.nfilters = nfilters
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.dense_layer_width = dense_layer_width
        self.dense_layer_width2 = dense_layer_width2
        self.use_res = use_res

        mel_scale = 'htk'
        all_freqs = torch.linspace(0, sample_rate // 2, nfreq)
        # calculate mel freq bins
        m_min = self._hz_to_mel(0, mel_scale=mel_scale)
        m_max = self._hz_to_mel(sample_rate/2.0, mel_scale=mel_scale)

        m_pts = torch.linspace(m_min, m_max, self.nfilters + 2)
        f_pts = self._mel_to_hz(m_pts, mel_scale=mel_scale)
        self.bounds = [0,]
        for freq_inx in range(1, len(f_pts)-1):
            self.bounds.append((all_freqs > f_pts[freq_inx]).float().argmax().item())
        self.bounds.append(nfreq)
        # print(self.bounds)
        self.trans1 = nn.ModuleList()
        self.trans2 = nn.ModuleList()
        self.trans3 = nn.ModuleList()
        for freq_inx in range(self.nfilters):
            self.trans1.append(nn.Linear((self.bounds[freq_inx+2]-self.bounds[freq_inx])*(self.n_channels+1), self.dense_layer_width, bias=False))
            self.trans2.append(nn.Conv1d(self.dense_layer_width, (self.bounds[freq_inx+2]-self.bounds[freq_inx])*self.dense_layer_width2, 1))
            if(self.use_res):
                self.trans3.append(nn.Conv2d(self.n_channels,self.dense_layer_width2, 1, bias=False))
        # print(self.trans1)
        # print(self.trans2)
            
    def _hz_to_mel(self, freq: float, mel_scale: str = "htk") -> float:
        r"""Convert Hz to Mels.

        Args:
            freqs (float): Frequencies in Hz
            mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

        Returns:
            mels (float): Frequency in Mels
        """

        if mel_scale not in ["slaney", "htk"]:
            raise ValueError('mel_scale should be one of "htk" or "slaney".')

        if mel_scale == "htk":
            return 2595.0 * math.log10(1.0 + (freq / 700.0))

        # Fill in the linear part
        f_min = 0.0
        f_sp = 200.0 / 3

        mels = (freq - f_min) / f_sp

        # Fill in the log-scale part
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0

        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz) / logstep

        return mels
    
    def _mel_to_hz(self, mels: torch.Tensor, mel_scale: str = "htk") -> torch.Tensor:
        """Convert mel bin numbers to frequencies.

        Args:
            mels (torch.Tensor): Mel frequencies
            mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

        Returns:
            freqs (torch.Tensor): Mels converted in Hz
        """

        if mel_scale not in ["slaney", "htk"]:
            raise ValueError('mel_scale should be one of "htk" or "slaney".')

        if mel_scale == "htk":
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0

        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

        return freqs
    
    def power_to_db(self, input):
        r"""Power to db, this function is the pytorch implementation of 
        librosa.power_to_lb
        """
        ref_value = 1.0
        amin = 1e-10
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

        top_db = 80
        if top_db < 0:
            raise librosa.util.exceptions.ParameterError('top_db must be non-negative')
        log_spec = torch.clamp(log_spec, min=log_spec.max().item() - top_db, max=np.inf)

        return log_spec
    
    def forward(self, x, inverse):
        if(inverse):
            # B C T F
            if(self.use_res):
                res, x = x
            out = torch.zeros([x.shape[0],self.dense_layer_width2,self.nfreq,x.shape[2]], dtype=x.dtype, layout=x.layout, device=x.device)
            for freq_inx in range(self.nfilters):
                out[:,:,self.bounds[freq_inx]:self.bounds[freq_inx+2],:] = out[:,:,self.bounds[freq_inx]:self.bounds[freq_inx+2],:] + \
                    self.trans2[freq_inx](x[:,:,:,freq_inx]).reshape(x.shape[0],self.dense_layer_width2,-1,x.shape[-2])
            out[:,:,self.bounds[1]:self.bounds[-2],:] = out[:,:,self.bounds[1]:self.bounds[-2],:] / 2.0
            out = out.permute(0,1,3,2).contiguous()
            if(self.use_res):return out + res
            else: return out
        else:  
            x = self.power_to_db(x)
            x = x.reshape(x.shape[0],self.n_channels, *x.shape[-2:]) # B C T F
            if(self.use_res):
                res = torch.cat([self.trans3[freq_inx](x[:,:,:,self.bounds[freq_inx]:self.bounds[freq_inx+2]]).tanh() \
                    for freq_inx in range(self.nfilters)],-1) # B T C F
            else:
                res = None
            x = x.permute(0,2,1,3).contiguous() # B T C F
            x = torch.cat([x,(x[:,:,0::2,:].pow(2.0)+1e-8).log()],-2)
            # for freq_inx in range(self.nfilters):
            #     print(x[:,:,:,self.bounds[freq_inx]:self.bounds[freq_inx+2]].flatten(start_dim=2).shape, self.trans1[freq_inx])
            x = torch.stack([self.trans1[freq_inx](x[:,:,:,self.bounds[freq_inx]:self.bounds[freq_inx+2]].flatten(start_dim=2)) \
                for freq_inx in range(self.nfilters)],-1) # B T C F
            x = x.permute(0,2,1,3).contiguous()
            if(self.use_res):
                return res, x
            else:
                return x
            
# test learn ov mel
# m = LearnOVSTFT2MELChannel(513,64,1,32,1,False,32000)
# inp = torch.rand(20,1,1001,513) # B C T F
# print('inp:',inp.shape)
# out = m(inp, False)
# print('out:',out.shape)

# out = m(out, True)
# print('recover:',out.shape)

# inp: torch.Size([20, 1, 1001, 513])
# out: torch.Size([20, 32, 1001, 64])
# recover: torch.Size([20, 1, 1001, 513])