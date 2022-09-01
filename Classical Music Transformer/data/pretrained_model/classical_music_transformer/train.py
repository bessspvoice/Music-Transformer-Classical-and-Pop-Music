from model import *
import os
import time

def train_model(epoch, model, optimizer, scheduler, train_x, train_y, batch_size):
    model.train() # Turn on the train mode
    CELoss = nn.CrossEntropyLoss()
    BCELoss = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    total_loss = 0.0
    tmp_loss = 0.0
    start_time = time.time()

    batches = np.arange(train_x.shape[0]//batch_size + 1)
    np.random.shuffle(batches)
    for b in range(len(batches)):
        batch = batches[b]
        if batch == train_x.shape[0]//batch_size:
            batch_x = torch.from_numpy(train_x[-batch_size:, :, :-24]).type(torch.FloatTensor).cuda()
            batch_y = torch.from_numpy(train_y[-batch_size:, :, :-24]).type(torch.FloatTensor).cuda()
            # tone_y = torch.from_numpy(train_y[-batch_size:, :, -24:-12]).type(torch.FloatTensor).cuda()
            chord_y = torch.from_numpy(train_y[-batch_size:, :, -12:]).type(torch.FloatTensor).cuda()

        else:
            batch_x = torch.from_numpy(train_x[batch*batch_size:(batch+1)*batch_size, :, :-24]).type(torch.FloatTensor).cuda()
            batch_y = torch.from_numpy(train_y[batch*batch_size:(batch+1)*batch_size, :, :-24]).type(torch.FloatTensor).cuda()
            # tone_y = torch.from_numpy(train_y[batch*batch_size:(batch+1)*batch_size, :, -24:-12]).type(torch.FloatTensor).cuda()
            chord_y = torch.from_numpy(train_y[batch*batch_size:(batch+1)*batch_size, :, -12:]).type(torch.FloatTensor).cuda()

        optimizer.zero_grad()

        mask = create_mask(batch_x)
        batch_type = batch_x[:,:,0:5]
        batch_barpos = batch_x[:,:,5:39]
        batch_tone = batch_x[:,:,39:64]
        batch_chord = batch_x[:,:,64:173]
        batch_dur = batch_x[:,:,173:238]
        batch_pitch = batch_x[:,:,238:367]

        h, pred_type = model.forward_hidden(batch_type, batch_barpos, batch_tone, batch_chord, batch_dur, batch_pitch, mask)
        pred_barpos, pred_tone, pred_chord, pred_dur, pred_pitch = model.forward_output(h, batch_y)

        pred_type = pred_type.view(pred_type.shape[0]*pred_type.shape[1], pred_type.shape[2])
        pred_barpos = pred_barpos.view(pred_barpos.shape[0]*pred_barpos.shape[1], pred_barpos.shape[2])
        pred_tone = pred_tone.view(pred_tone.shape[0]*pred_tone.shape[1], pred_tone.shape[2])
        pred_chord = pred_chord.view(pred_chord.shape[0]*pred_chord.shape[1], pred_chord.shape[2])
        pred_dur = pred_dur.view(pred_dur.shape[0]*pred_dur.shape[1], pred_dur.shape[2])
        pred_pitch = pred_pitch.view(pred_pitch.shape[0]*pred_pitch.shape[1], pred_pitch.shape[2])
        batch_y = batch_y.view(batch_y.shape[0]*batch_y.shape[1], batch_y.shape[2])
        # tone_y = tone_y.view(tone_y.shape[0]*tone_y.shape[1], tone_y.shape[2])
        chord_y = chord_y.view(chord_y.shape[0]*chord_y.shape[1], chord_y.shape[2])

        # type
        y_type = batch_y[:, 0:5]
        y_barpos = batch_y[:, 5:39]
        y_tone = batch_y[:, 39:64]
        y_chord = batch_y[:, 64:173]
        y_dur = batch_y[:, 173:238]
        y_pitch = batch_y[:, 238:367]

        CE_type = CELoss(pred_type, torch.argmax(y_type, dim=1))
        CE_barpos = CELoss(pred_barpos, torch.argmax(y_barpos, dim=1))
        CE_tone = CELoss(pred_tone, torch.argmax(y_tone, dim=1))
        CE_chord = CELoss(pred_chord, torch.argmax(y_chord, dim=1))
        CE_dur = CELoss(pred_dur, torch.argmax(y_dur, dim=1))
        CE_pitch = CELoss(pred_pitch, torch.argmax(y_pitch, dim=1))

        index = (y_pitch[:,0] == 0)
        ### tonality loss ###
        # tone_y = torch.cat([tone_y, tone_y, tone_y, tone_y, tone_y, tone_y, tone_y, tone_y, tone_y, tone_y, tone_y[:, :8]], dim=-1)
        # BCE_tone = BCELoss(sigmoid(pred_pitch[index, 1:]), tone_y[index])

        ### chord loss ###
        chord_y = torch.cat([chord_y, chord_y, chord_y, chord_y, chord_y, chord_y, chord_y, chord_y, chord_y, chord_y, chord_y[:, :8]], dim=-1)
        BCE_chord = BCELoss(sigmoid(pred_pitch[index, 1:]), chord_y[index])

        loss = torch.mean(CE_type + CE_barpos + CE_tone + CE_chord + CE_dur + CE_pitch + BCE_chord)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        tmp_loss += loss.item()

        log_interval = 200
        if b % log_interval == 0 and b > 0:
            cur_loss = tmp_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, b, train_x.shape[0]//batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))

            tmp_loss = 0
            start_time = time.time()

    total_loss /= (train_x.shape[0])
    print('====> Epoch: {:3d} Train Average loss: {:.6f}'.format(epoch, total_loss))
    
    return total_loss

### GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

## load dataset
train = np.load("train_BPSFH.npy")

### make into sequences of 128, with slide window 1
train_slide = np.transpose(strided_axis1(train.T, 128, 1), (0,2,1))
train_x = train_slide[:-1]
train_y = train_slide[1:]

### model settings
d_type = 5
d_barpos = 34
d_tone = 25
d_chord = 109
d_dur = 65
d_pitch = 129
d_attention = 512
heads = 8
N = 12
model = Transformer(d_type, d_barpos, d_tone, d_chord, d_dur, d_pitch, d_attention, N, heads)
if torch.cuda.is_available():
    model.cuda()
        
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

epochs = 5
batch_size = 16
history_train = []

for epoch in range(0, epochs + 1):
    epoch_start_time = time.time()
    train_loss = train_model(epoch, model, optimizer, scheduler, train_x, train_y, batch_size)
    history_train.append([train_loss])
    
    np.save("./history_train.npy", np.array(history_train))
    print("epoch", epoch+1, "saving model.")
    torch.save(model.state_dict(), "./"+str(epoch+1)+".pt")
    
    scheduler.step()
