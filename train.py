import json
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import argparse
import numpy as np
from typesql.utils import *
from typesql.model.sqlnet import SQLNet
# Keaton

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true',
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--suffix', type=str, default='',
            help='The suffix at the end of saved model name.')
    parser.add_argument('--sd', type=str, default='',
            help='set model save directory.')
    parser.add_argument('--db_content', type=int, default=0,
            help='0: use knowledge graph type, 1: use db content to get type info')
    parser.add_argument('--train_emb', action='store_true',
            help='Train word embedding.')
    parser.add_argument('--log_dir', type=str, default='./logs', help="log directory")
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    N_word=600
    B_word=42
    if args.toy:
        print('toy')
        USE_SMALL=True
        GPU=True
        BATCH_SIZE=5
    else:
        USE_SMALL=False
        GPU=True
        BATCH_SIZE=256
    TRAIN_ENTRY=(True, True, True)  # (AGG, SEL, COND)
    TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY
    learning_rate = 5e-3

    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(use_small=USE_SMALL)

    #word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
    #        load_used=args.train_emb, use_small=USE_SMALL)
    if args.db_content == 0:
        word_emb = load_word_and_type_emb('glove/glove.42B.300d.txt', "para-nmt-50m/data/paragram_sl999_czeng.txt",\
                                            val_sql_data, val_table_data, args.db_content, is_list=True, use_htype=False)
    else:
        word_emb = load_concat_wemb('glove/glove.42B.300d.txt', "para-nmt-50m/data/paragram_sl999_czeng.txt")

    model = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0)

    agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_name(args)

    if args.train_emb: # Load pretrained model.
        agg_lm, sel_lm, cond_lm = best_model_name(args, for_load=True)
        print(f"Loading from {agg_lm}")
        model.agg_pred.load_state_dict(torch.load(agg_lm))
        print(f"Loading from {sel_lm}")
        model.selcond_pred.load_state_dict(torch.load(sel_lm))
        print(f"Loading from {cond_lm}")
        model.cond_pred.load_state_dict(torch.load(cond_lm))


    #initial accuracy
    init_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY, args.db_content)
    best_agg_acc = init_acc[1][0]
    best_agg_idx = 0
    best_sel_acc = init_acc[1][1]
    best_sel_idx = 0
    best_cond_acc = init_acc[1][2]
    best_cond_idx = 0

    print(f"Init dev acc_qm {init_acc[0]}\n breakdown on (agg, sel, where): {init_acc[1]}")
    #print 'Init dev acc_qm: %s\n  breakdown on (agg, sel, where): %s' % init_acc
    if TRAIN_AGG:
        torch.save(model.agg_pred.state_dict(), agg_m)
        torch.save(model.agg_type_embed_layer.state_dict(), agg_e)
    if TRAIN_SEL:
        torch.save(model.selcond_pred.state_dict(), sel_m)
        torch.save(model.sel_type_embed_layer.state_dict(), sel_e)
    if TRAIN_COND:
        torch.save(model.op_str_pred.state_dict(), cond_m)
        torch.save(model.cond_type_embed_layer.state_dict(), cond_e)


    writer = SummaryWriter(log_dir=args.log_dir)
    for i in range(args.epochs):
        #print 'Epoch %d @ %s'%(i+1, datetime.datetime.now())
        print(f'Epoch {i+1} @ {datetime.datetime.now()}')
        loss = epoch_train(model, optimizer, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY, args.db_content)
        print(f' Loss = {loss}')
        train_acc, train_entry_acc = epoch_acc(model, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY, args.db_content)
        print(f' Train acc_qm: {train_acc}\n breakdown result: {train_entry_acc}')
        val_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY, args.db_content, False) #for detailed error analysis, pass True to the end
        print(f' Dev acc_qm: {val_acc}\n breakdown result: {val_acc[1]}')
        writer.add_scalar('loss', loss, i)
        writer.add_scalar('train_acc_qm', train_acc, i)
        writer.add_scalar('val_acc_qm', val_acc[0], i)
        if TRAIN_AGG:
            if val_acc[1][0] > best_agg_acc:
                best_agg_acc = val_acc[1][0]
                best_agg_idx = i+1
                torch.save(model.agg_pred.state_dict(),
                    args.sd + '/epoch%d.agg_model%s'%(i+1, args.suffix))
                torch.save(model.agg_pred.state_dict(), agg_m)

            torch.save(model.agg_type_embed_layer.state_dict(),
                                args.sd + '/epoch%d.agg_embed%s'%(i+1, args.suffix))
            torch.save(model.agg_type_embed_layer.state_dict(), agg_e)

        if TRAIN_SEL:
            if val_acc[1][1] > best_sel_acc:
                best_sel_acc = val_acc[1][1]
                best_sel_idx = i+1
                torch.save(model.selcond_pred.state_dict(),
                    args.sd + '/epoch%d.sel_model%s'%(i+1, args.suffix))
                torch.save(model.selcond_pred.state_dict(), sel_m)

                torch.save(model.sel_type_embed_layer.state_dict(),
                                args.sd + '/epoch%d.sel_embed%s'%(i+1, args.suffix))
                torch.save(model.sel_type_embed_layer.state_dict(), sel_e)

        if TRAIN_COND:
            if val_acc[1][2] > best_cond_acc:
                best_cond_acc = val_acc[1][2]
                best_cond_idx = i+1
                torch.save(model.op_str_pred.state_dict(),
                    args.sd + '/epoch%d.cond_model%s'%(i+1, args.suffix))
                torch.save(model.op_str_pred.state_dict(), cond_m)

                torch.save(model.cond_type_embed_layer.state_dict(),
                                args.sd + '/epoch%d.cond_embed%s'%(i+1, args.suffix))
                torch.save(model.cond_type_embed_layer.state_dict(), cond_e)

        print(f" Best val acc = {(best_agg_acc, best_sel_acc, best_cond_acc)}, on epoch {(best_agg_idx, best_sel_idx, best_cond_idx)} individually")
        #print ' Best val acc = %s, on epoch %s individually'%(
        #        (best_agg_acc, best_sel_acc, best_cond_acc),
        #        (best_agg_idx, best_sel_idx, best_cond_idx))
    writer.close()
