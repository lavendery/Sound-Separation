# Ke Chen
# knutchen@ucsd.edu
# Zero-shot Audio Source Separation via Query-based Learning from Weakly-labeled Data
# The configuration file

# for model training
exp_name = "separation-from-sed-condition-grad_256" # the saved ckpt prefix name of the model 
input_ov_stft2melchannel = True
input_ov_stft2complex = False


workspace = "" # the folder of your code
dataset_path = "./audioset" # the dataset path
index_type = "full_train"
idc_path = "data/idc" # the folder of audioset class count files
balanced_data = True

# trained from a checkpoint, or evaluate a single model 
resume_checkpoint = None
sed_resume_checkpoint = "./HTSAT_AudioSet_Saved_0.ckpt"
separation_resume_checkpoint = None


loss_type = "mae"
loss_sisdr = True
sed_loss_type = "clip_bce" # AudioSet & SCV2: "clip_bce" |  ESC-50: "clip_ce" 
classes_num = 527 # esc: 50 | audioset: 527 | scv2: 35
dataset_type = "audioset" # "audioset" "esc-50" "scv2"

gather_mode = False
debug = False

classes_num = 527
eval_list = [] # left blank to preserve all classes, otherwise will filter the specified classes
# [15, 63, 81, 184, 335, 449, 474, 348, 486, 4] # randomly generated from the 527-classes for held-out evaludation


# batch_size = 16 * 8   # batch size per GPU x GPU number , default is 16 x 8 = 128
batch_size = 4 * 8
learning_rate = 1e-3 # 3e-4 is also workable
max_epoch = 1000
num_workers = 3

# lr_scheduler_epoch = [90, 110]
lr_scheduler_epoch = [10,20,30]
lr_rate = [0.05, 0.1, 0.2]

latent_dim = 2048

# for signal processing
sample_rate = 32000
clip_samples = sample_rate * 10 # audio_set 10-sec clip
segment_frames = 200 
hop_samples = 320
random_seed = 12412 # 444612 1536123 12412
random_mode = "one_class" # "no_random, one_class, random, order", one class is the best
window_size = 1024
mel_bins = 64

# for evaluation
musdb_path = "/home/Research/ZS_ASP/data/musdb-wav/" # musdb download folder
testavg_path = "dataset/MUSDB/musdb30-train-32000fs.npy" # the processed training set (to get the latent query)
testset_path = "dataset/MUSDB/musdb-test-32000fs.npy" # the processed testing set (to calculate the performance)
test_key = ["vocals", "drums", "bass", "other"] # four tracks for musdb, and your named track for other inference
test_type = "mix"
infer_type = "mean"
energy_thres = 0.1
wave_output_path = "/home/Research/ZS_ASP/wavoutput" # output folder
using_wiener = True # use wiener filter or not (default: True)
using_whiting = False # use whiting or not (default: False)

# weight average
wa_model_folder = "/home/Research/ZS_ASP/version_3/checkpoints/"
wa_model_path = "zs_wa.ckpt"

# for inference
inference_file = "/home/Research/ZS_ASP/data/pagenini.wav" # an audio file to separate
inference_query = "/home/Research/ZS_ASP/data/query" # a folder containing all samples for obtaining the query
overlap_rate = 0.0 # [0.0, 1.0), 0 to disabled, recommand 0.5 for 50% overlap. Overlap will increase computation time and improve result quality
