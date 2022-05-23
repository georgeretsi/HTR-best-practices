
#classes = '_!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '


#cdict = {c:i for i,c in enumerate(classes)}
#icdict = {i:c for i,c in enumerate(classes)}

k = 1
cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]

head_cfg = (256, 3)  # (hidden , num_layers)

#head_type = 'rnn'

flattening='maxpool'
#flattening='concat'

stn=False

max_epochs = 240

batch_size = 20
level = "line"
fixed_size = (4 * 32, 4 * 256)

#batch_size = 100
#level = "word"
#fixed_size = (1 * 64, 256)

save_path = './saved_models/'
load_code = None