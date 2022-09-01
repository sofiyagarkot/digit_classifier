__author__ = 'Maayan'

from brian2 import *

global units_list
units_list = {'e_e': mV, 'e_i': mV, 'e_t': mV, 'e_leak': mV, 'g_t': nsiemens, 'g_leak': nsiemens, 'capacit': pF,
                  'v_thresh': mV, 'v_reset': mV, 'refract': ms, 'tau_m': ms, 'tau_e': ms, 'tau_i': ms, 'i_ext': pA}


# parameters
# connectivity
global n_e, n_i   # number of excitatory and inhibitory neurons
n_e = 4000*1
n_i = 1000*1

global p_ee, p_ei, p_ii, p_ie, adj_flag, topology
p_ee = 0.12*1
p_ei = 0.20*1
p_ii = 0.24*1
p_ie = 0.22*1
adj_flag = True
topology = 'random'

global w_ee_dist, w_ee_mu, w_ee_sigma, w_ei_dist, w_ei_mu, w_ei_sigma, w_ii_dist, w_ii_mu, w_ii_sigma, w_ie_dist, w_ie_mu, w_ie_sigma
w_ee_dist = 'lognormal'
w_ee_mu = -0.6*1
w_ee_sigma = 0.5*1
w_ei_dist = 'lognormal'
w_ei_mu = -0.6*1
w_ei_sigma = 0.5*1
w_ii_dist = 'lognormal'
w_ii_mu = -0.6*1
w_ii_sigma = 0.5*1
w_ie_dist = 'lognormal'
w_ie_mu = -0.6*1
w_ie_sigma = 0.5*1

# electrophys
global e_e, e_i, e_t, e_leak, g_t, g_leak, capacit, v_thresh, v_reset, refract, tau_m, tau_e, tau_i, scale_inh, i_ext, d_ge
e_e = 0*mV
e_i = -80*mV
e_t = 0*mV
e_leak = -65*mV
g_t = 30*nsiemens
g_leak = 10*nsiemens
capacit = 230*pF
v_thresh = -50*mV
v_reset = -65*mV
refract = 3*ms
tau_m = 20*ms
tau_e = 5*ms
tau_i = 10*ms
scale_inh = 1.2*1
i_ext = 0*pA
d_ge = 0.6*nsiemens

# plasticity
global plast_flag, shift, tau_pre, tau_post, amp_depress, amp_potent, max_w, min_w
plast_flag = 'plast_on'
shift = 2*ms
tau_pre = 20*ms
tau_post = 20*ms
amp_depress = -0.008*nsiemens
amp_potent = 0.006*nsiemens
max_w = 10*nsiemens
min_w = 0*nsiemens

global min_delt, clv
min_delt = -10000 * ms
clv = 0 * ms


# equations
global lif_eqs, syn_eqs, syn_on_pre_ex, syn_on_pre_inh, stdp_eq, stdp_eq_freeze, syn_on_post
lif_eqs = '''
    dv/dt = (g_leak*(e_leak-v)+ge*(e_e-v)+gi*(e_i-v)+i_ext)/capacit : volt (unless refractory)
    dge/dt = -ge/tau_e : siemens
    dgi/dt = -gi/tau_i : siemens
    '''

syn_eqs = {'plast_on': '''
               weight_e : siemens
               weight_i : siemens
               p_allowed : 1
               pre_time: second
               post_time: second
               ''', 'plast_off': '''
               weight_e : siemens
               weight_i : siemens
               '''}

syn_on_pre_ex = {'plast_on': '''
                  ge += weight_e
                  pre_time = t
                  weight_e = clip(weight_e+p_allowed*amp_depress*exp(((post_time-pre_time)-shift)/tau_post), min_w, max_w)                                     
                  ''', 'plast_off': '''
                  ge += weight_e
                  '''}

syn_on_pre_inh = {'plast_on': '''
                  gi += weight_i
                  pre_time = t
                  weight_i = clip(weight_i+p_allowed*amp_depress*exp(((post_time-pre_time)-shift)/tau_post), min_w, max_w)                                     
                  ''', 'plast_off': '''
                  gi += weight_i
                  '''}

# LTP or LTD here (depending on the shift)
# condition 1 is for LTP, and condition 2 for LTD
# conditions are int()
stdp_eq = "clip(weight_e+int((post_time-pre_time)>shift)"\
           "*p_allowed*amp_potent*exp(((pre_time-post_time)+shift)/tau_pre)+int((post_time-pre_time)<=shift)*" \
           "p_allowed*amp_depress*exp(clip((post_time-pre_time)-shift, min_delt, clv)/tau_post), min_w, max_w)"

stdp_eq_freeze = "clip(weight_e+int((post_time-pre_time)>shift)"\
           "*p_allowed*amp_potent*exp(((pre_time-post_time)+shift)/tau_pre)" \
           ", min_w, max_w)"

syn_on_post = {'plast_on': f'''
                    post_time = t
                    weight_e = {stdp_eq}                
                   ''', 'plast_off': '''
                   weight_e = weight_e
                   '''}

class Config:
    def __init__(self, args):
        self.N = args.N
        self.path_to_mappings = args.path_to_mappings
        self.path_to_results = args.path_to_results
        self.path_to_visuals = args.path_to_visuals

        self.precomputed_mappings = False
