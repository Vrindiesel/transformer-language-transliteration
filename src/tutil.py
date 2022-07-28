"""
Created by davan 
7/26/22
"""
import util

def print_examples(i2c, N, loss_mask, pred, src, trg):

    if pred is None:
        u_pred = [[str(None)]] * N
    else:
        u_pred = util.unpack_batch(pred)

    u_trg = util.unpack_batch(trg)
    u_src = util.unpack_batch(src)
    u_lm = util.unpack_batch(loss_mask)
    print("Some Examples")
    for ss, p, t, lm in zip(u_src[:N], u_pred[:N], u_trg[:N], u_lm[:N]):
        print("-" * 10)
        print("input:", [i2c[j] for j in ss])
        print("targ:", [i2c[int(j)] for j in t])
        print("pred:", None if pred is None else " ".join([i2c[j] for j in p]) )
        print("LM:", lm)
    input(">>>")



