function sample = get_epoch_data(record, epoch, Fs)

sample = record(((epoch-1)*Fs*30+1) : epoch*Fs*30);

end