import sys
import argparse
import os

def load_raw_data(filepath):
    tokens = []
    sents = []

    with open(filepath, "r") as f:
        for line in f:
            token = line.rstrip()

            if (token == "") or (token == " "):
                sents.append(tokens)
                tokens = []
            else:
                tokens.append(token)

    return sents

def load_conllu_data(filepath):
    tokens = []
    sents = []

    with open(filepath, "r") as f:
        for line in f:
            if line[0] == '#':
                continue
            line_data = line.rstrip().split("\t")

            if line_data == [''] or line_data == [" "] or line_data == []:
                sents.append(tokens)
                tokens = []
            else:
                if line_data[0].startswith("# sent_id =") or line_data[0].startswith("# text ="):
                    pass
                else:
                    t_text = line_data[1].lower()
                    t_pos = line_data[3]
                    t_head = line_data[6]
                    t_rel = line_data[7]
                    tokens.append([t_text, t_head, t_rel, t_pos])

    return sents

def evaluate(raw_pred, raw_gold, pred, gold):
    # Ensure the number of sentences in raw pred and raw gold are the same
    if len(raw_pred) != len(raw_gold):
        sys.exit("ERROR: The predicted raw file has a different number of sentences (pred: {}, gold: {}). Exiting...".format(
            len(raw_pred), len(raw_gold)))

    pos_correct = 0
    uas_correct = 0
    las_correct = 0
    total_gold_tokens = 0
    splits_counter = 0

    # Iterate over raw and gold sentences
    for sent_pred, sent_gold, pred, gold in zip(raw_pred, raw_gold, pred, gold):
        # Ensure the number of tokens in raw pred and raw gold are the same
        if len(sent_pred) != len(sent_gold):
            sys.exit("ERROR: A sentence in the predicted raw file has a different token length (pred sent: {}). Exiting...".format(
                ' '.join(t for t in sent_pred)))

        token_id = 0
        pred_multitokens = []

        # Iterate over raw pred and gold sentence tokens, and detect and store misalignments
        # e.g., raw_pred: sentire si / raw_gold: sentirsi / pred: sentire\nsi / gold: sentirsi
        for word_pred, word_gold in zip(sent_pred, sent_gold):
            # Predicted multitokens have (at least) a space in it
            spaces_count = word_pred.count(" ")
            
            # Store ids and space counts of multitokens
            if spaces_count > 0:
                pred_multitokens.append([token_id, spaces_count])
            token_id += 1

        # Merge and align the predicted head indices
        if len(pred_multitokens) > 0:
            for pred_multitoken in pred_multitokens:
                pred_multitoken_id, pred_multitoken_count = pred_multitoken

                # Merge extra tokens into the first one
                curr_subtoken = pred[pred_multitoken_id]
                for itemIdx in range(len(pred)):
                    pred[itemIdx][1] = str(pred[itemIdx][1])
                for i in range(1, pred_multitoken_count+1):
                    next_subtoken = pred[pred_multitoken_id+i]
                    curr_subtoken[0] = curr_subtoken[0] + "_SEP_" + next_subtoken[0]
                    curr_subtoken[1] = curr_subtoken[1] + "_SEP_" + next_subtoken[1]
                    curr_subtoken[2] = curr_subtoken[2] + "_SEP_" + next_subtoken[2]
                    curr_subtoken[3] = curr_subtoken[3] + "_SEP_" + next_subtoken[3]

                # Remove the extra tokens
                for i in range(1, pred_multitoken_count+1):
                    del pred[pred_multitoken_id+1]

                # Update the predicted head indices if these are greater than the first one, taking care of multitoken references
                for pred_token in pred:
                    indices_to_align = [i for i in range(
                        pred_multitoken_id+1, pred_multitoken_id+pred_multitoken_count+1)]

                    head_ids = pred_token[1].split("_SEP_")
                    #if head_ids == ['_']:#TODO why?
                
                    if len(head_ids) == 1:
                        if int(head_ids[0]) > pred_multitoken_id:
                            if int(head_ids[0]) in indices_to_align:
                                pred_token[1] = pred_multitoken_id
                            else:
                                pred_token[1] = str(int(pred_token[1]) - 1)
                    elif len(head_ids) > 1:
                        new_head = ""
                        for head_id in head_ids:
                            if new_head != "":
                                new_head += "_SEP_"
                            if int(head_id) > pred_multitoken_id:
                                if int(head_id) in indices_to_align:
                                    new_head = str(pred_multitoken_id)
                                else:
                                    new_head += str(int(head_id) - 1)
                            else:
                                new_head += head_id
                        pred_token[1] = str(new_head)
                    else:
                        sys.exit("ERROR: The head token index is empty")

        token_id = 0

        # Iterate over pred and gold sentence tokens for alignment-aware evaluation
        for word_pred, word_gold in zip(pred, gold):
            # If a token is part of a multitoken, keep track of the offset
            #print(token_id, word_pred, word_gold)

            pos_correct += compute_token_pos(word_pred, word_gold)
            uas_correct += compute_token_uas(word_pred, word_gold)
            las_correct += compute_token_las(word_pred, word_gold)
            total_gold_tokens += 1

            token_id += 1

        # Reinitialize the list of indices of predicted misalignments
        if len(pred_multitokens) > 0:
            for pred_multitoken in pred_multitokens:
                splits_counter += pred_multitoken[1]
        else:
            splits_counter += 0
        pred_multitokens = []
        #print()

    return uas_correct/total_gold_tokens*100, las_correct/total_gold_tokens*100, splits_counter, pos_correct/total_gold_tokens*100

    # Print the evaluation results
    print("EVALUATION RESULTS:")
    print("===================")
    print("UAS score: " + format(uas_correct/total_gold_tokens*100, ".2f") + "%" + "\t(" + str(uas_correct) + "/" + str(total_gold_tokens) + ")")
    print("LAS score: " + format(las_correct/total_gold_tokens*100, ".2f") + "%" + "\t(" + str(las_correct) + "/" + str(total_gold_tokens) + ")")
    print("===================")
    print("NOTES: The system has produced {} split(s)".format(splits_counter))

def compute_token_pos(pred, gold):
    pred_pos_tags = pred[3]
    gold_pos_tag = gold[3]

    #if "_SEP_" in pred_pos_tags:
    #    print(pred_pos_tags, gold_pos_tag)

    preds_p = pred_pos_tags.split("_SEP_")
    if len(preds_p) > 1:
        for pred_pos_tag in preds_p:
            if pred_pos_tag == gold_pos_tag:
                return 1
        return 0
    else:
        if (preds_p[0] == gold_pos_tag):
            return 1
        else:
            return 0

def compute_token_uas(pred, gold, strategy="at-least-one"):
    pred_heads = str(pred[1])
    gold_head = gold[1]

    if strategy=="at-least-one":
        preds_h = pred_heads.split("_SEP_")
        if len(preds_h) > 1:
            for pred_head in preds_h:
                if pred_head == gold_head:
                    return 1
            return 0
        else:
            if (preds_h[0] == gold_head):
                return 1
            else:
                return 0

def compute_token_las(pred, gold, strategy="at-least-one"):
    pred_heads = str(pred[1])
    pred_rels = pred[2]
    gold_head = str(gold[1])
    gold_rel = gold[2]

    if strategy=="at-least-one":
        preds_h = pred_heads.split("_SEP_")
        preds_r = pred_rels.split("_SEP_")
        if (len(preds_h) > 1) and (len(preds_r) > 1):
            for i in range(len(preds_h)):
                if (preds_h[i] == gold_head) and (preds_r[i] == gold_rel):
                    return 1
            return 0
        else:
            if (preds_h[0] == gold_head) and (preds_r[0] == gold_rel):
                return 1
            else:
                return 0
    else:
        sys.exit("ERROR: The strategy has not been implemented yet")


def makeTable(metric):
    treebanks = sorted([x.replace('.conllu', '') for x in os.listdir('treebanks')])
    # Hardcoded removal of en-aae from the score computation for POS tagging (theres no annotation!)
    if metric == 'upos':
        treebanks.remove("ud-en-aae.test")
    allScores = []
    for team in os.listdir('../submissions'):
        #print(team)
        teamScores = [[], [], team]
        for treebank in treebanks:
            raw_pred = load_raw_data('../submissions/' + team + '/extrinsic_evaluation/' + treebank + '.norm.pred')
            raw_gold = load_raw_data('treebanks-cleaned/' + treebank + '.txt')
            conllu_pred = load_conllu_data('../submissions/' + team + '/extrinsic_evaluation/' + treebank + '.norm.1.conllu.out')
            conllu_gold = load_conllu_data('treebanks-cleaned/' + treebank + '.conllu')
            #print('../submissions/' + team + '/extrinsic_evaluation/' + treebank + '.norm.1.conllu.out')
            uas, las, numSplits, pos_acc = evaluate(raw_pred, raw_gold, conllu_pred, conllu_gold)
            #teamScores[0].append(uas)
            if metric == 'uas':
                teamScores[0].append(uas)
            elif metric == 'las':
                teamScores[0].append(las)
            elif metric == 'upos':
                teamScores[0].append(pos_acc)
            teamScores[1].append(numSplits)
        teamScores[0].insert(0, sum(teamScores[0])/len(teamScores[0]))
        teamScores[1].insert(0, sum(teamScores[1])/len(teamScores[1]))
        allScores.append(teamScores)
    print(metric.upper())
    print(' & '.join(['treebank', 'avg.'] + [x.replace('.test', '').replace('ud-', '') for x in treebanks]) + ' \\\\')
    print('\\midrule')

    for treebankIdx in range(len(allScores[0][0])):
        highest = 0.0
        for teamIdx in range(len(allScores)):
            if allScores[teamIdx][0][treebankIdx] > highest:
                highest = allScores[teamIdx][0][treebankIdx]
        for teamIdx in range(len(allScores)):
            if allScores[teamIdx][0][treebankIdx] == highest:
                allScores[teamIdx][0][treebankIdx] = '\\textbf{' + '{:.2f}'.format(allScores[teamIdx][0][treebankIdx]) + '}'
            else:
                allScores[teamIdx][0][treebankIdx] = '{:.2f}'.format(allScores[teamIdx][0][treebankIdx])
            allScores[teamIdx][0][treebankIdx] = allScores[teamIdx][0][treebankIdx] + '-' + str(int(allScores[teamIdx][1][treebankIdx]))

    for row in sorted(allScores, reverse=True):
        tableRow = [row[2]] + row[0]
        if 'CL-MoNoise' in tableRow[0] or 'MaChAmp' in tableRow[0] or 'HEL-LJU' in tableRow[0]:
            tableRow[0] += '$^*$'
        if tableRow[0] in ['MoNoise', 'MFR', 'LAI']:
            print('\\rowcolor{Gray}')
        print(' & '.join(tableRow) + ' \\\\')
    print()

if __name__ == '__main__':

    for metric in 'uas', 'las', 'upos':
        makeTable(metric)
    
