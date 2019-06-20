import tensorflow as tf
from tensorflow.contrib.layers.python.layers import fully_connected
from dataset_utils import DatasetLoader

def add_fc(inputs, outdim, train_phase, scope_in):
    fc =  fully_connected(inputs, outdim, activation_fn=None, scope=scope_in + '/fc')
    fc_bnorm = tf.layers.batch_normalization(fc, momentum=0.1, epsilon=1e-5,
                         training=train_phase, name=scope_in + '/bnorm')
    fc_relu = tf.nn.relu(fc_bnorm, name=scope_in + '/relu')
    fc_out = tf.layers.dropout(fc_relu, seed=0, training=train_phase, name=scope_in + '/dropout')
    return fc_out

def pdist(x1, x2):
    """
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1_square = tf.reshape(tf.reduce_sum(x1*x1, axis=1), [-1, 1])
    x2_square = tf.reshape(tf.reduce_sum(x2*x2, axis=1), [1, -1])
    return tf.sqrt(x1_square - 2 * tf.matmul(x1, tf.transpose(x2)) + x2_square + 1e-4)

def embedding_loss(im_embeds, sent_embeds, im_labels, args):
    """
        im_embeds: (b, 512) image embedding tensors
        sent_embeds: (sample_size * b, 512) sentence embedding tensors
            where the order of sentence corresponds to the order of images and
            setnteces for the same image are next to each other
        im_labels: (sample_size * b, b) boolean tensor, where (i, j) entry is
            True if and only if sentence[i], image[j] is a positive pair
    """
    # compute embedding loss
    #sent_im_ratio = args.sample_size
    #num_img = args.batch_size
    #num_sent = num_img * sent_im_ratio

    img_attr_ratio=args.sample_size
    num_attr=args.batch_size
    num_img=num_attr*img_attr_ratio
    #print(im_labels[0])
    #print(im_labels.shape)
    #print("img",im_embeds.shape)
    #print("sent",sent_embeds)


    sent_im_dist = pdist(im_embeds, sent_embeds)
    #print(sent_im_dist.shape)
    # attribute loss: image, positive attribute, and negative attribute
    pos_pair_dist = tf.reshape(tf.boolean_mask(sent_im_dist, im_labels), [num_img, 1])
    neg_pair_dist = tf.reshape(tf.boolean_mask(sent_im_dist, ~im_labels), [num_img, -1])
    im_loss = tf.clip_by_value(args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    im_loss = tf.reduce_mean(tf.nn.top_k(im_loss, k=args.num_neg_sample)[0])
    #im_loss = tf.reduce_mean(im_loss)
    # image loss: attribute, positive image, and negative image
    neg_pair_dist = tf.reshape(tf.boolean_mask(tf.transpose(sent_im_dist), ~tf.transpose(im_labels)), [num_attr, -1])
    neg_pair_dist = tf.reshape(tf.tile(neg_pair_dist, [1, img_attr_ratio]), [num_img, -1])
    sent_loss = tf.clip_by_value(args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    sent_loss = tf.reduce_mean(tf.nn.top_k(sent_loss, k=args.num_neg_sample)[0])
    #sent_loss = tf.reduce_mean(sent_loss)

    #pos_pair_dist = tf.reshape(tf.boolean_mask(tf.transpose(sent_im_dist), tf.transpose(im_labels)), [num_attr, -1])
    #pos_pair_dist = tf.reduce_max(pos_pair_dist, axis=1, keep_dims=False)
    #neg_pair_dist = tf.reshape(tf.boolean_mask(tf.transpose(sent_im_dist), ~tf.transpose(im_labels)), [num_attr, -1])
    #neg_pair_dist = tf.reduce_max(neg_pair_dist,axis=1,keep_dims=False)
    #sent_loss = tf.clip_by_value(args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    #sent_loss=tf.reduce_mean(tf.reduce_sum(sent_loss))

    # image only loss (neighborhood-preserving constraints)
    sent_sent_dist = pdist(im_embeds, im_embeds)
    sent_sent_mask = tf.reshape(tf.tile(tf.transpose(im_labels), [1, img_attr_ratio]), [num_img, num_img])
    pos_pair_dist = tf.reshape(tf.boolean_mask(sent_sent_dist, sent_sent_mask), [-1, img_attr_ratio])
    pos_pair_dist = tf.reduce_max(pos_pair_dist, axis=1, keep_dims=True)
    neg_pair_dist = tf.reshape(tf.boolean_mask(sent_sent_dist, ~sent_sent_mask), [num_img, -1])
    #neg_pair_dist = tf.reduce_min(neg_pair_dist, axis=1,keep_dims=False)
    sent_only_loss = tf.clip_by_value(args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    sent_only_loss = tf.reduce_mean(tf.nn.top_k(sent_only_loss, k=args.num_neg_sample)[0])
    #sent_only_loss = tf.reduce_mean(sent_only_loss)
    #sent_only_loss = tf.reduce_mean(tf.reduce_sum(sent_only_loss))



    #attribute loss

    #loss = tf.reduce_mean(pos_pair_dist)
    loss = im_loss * args.im_loss_factor + sent_loss + sent_only_loss * args.sent_only_loss_factor
    #loss = im_loss + sent_loss
    #loss = sent_loss
    #loss = im_loss * args.im_loss_factor + sent_only_loss * args.sent_only_loss_factor
    return loss


def recall_k(im_embeds, sent_embeds,im_labels, ks=None):
    """
        Compute recall at given ks.
    """
    count=0
    sent_im_dist = pdist(im_embeds, sent_embeds)
    #print("1",sent_im_dist.shape)
    #sent_sent_dist_1=pdist(attr1,sent_feats)
    #print("2",sent_sent_dist_1.shape)
    #sent_sent_dist_2=pdist(attr2,sent_feats)
    #dist=sent_im_dist+sent_sent_dist*0.01
    data_loader = DatasetLoader('/home/litongxin/image_attribute_two_branch/img_feat_test.mat',
                  '/home/litongxin/Two_branch_network/two_branch_img_feature.mat', split='eval')
    pred = tf.nn.top_k(-tf.transpose(sent_im_dist), k=1)[1]
    #for i in range(pred.shape[0]):
    #    im_id = data_loader.im_id[int(pred[i][0])][0]
    #    if im_id == data_loader.attr_test_feats.keys()[i] or \
    #            data_loader.attr_test_feats[im_id] == data_loader.attr_test_feats.values()[i]:
    #        count = count + 1.0
    #return count/data_loader.attr_test_feat_shape[0]
    return pred

def embedding_model(im_feats, sent_feats, train_phase, im_labels,
                    fc_dim = 1024, embed_dim = 512):
    """
        Build two-branch embedding networks.
        fc_dim: the output dimension of the first fc layer.
        embed_dim: the output dimension of the second fc layer, i.e.
                   embedding space dimension.
    """
    # Image branch.
    #is_training = tf.placeholder(dtype=tf.bool, shape=())
    im_fc1 = add_fc(im_feats, fc_dim, train_phase, 'im_embed_1')
    #im_fc2=add_fc(im_fc1,512,train_phase,'im_embed_2')
    #im_fc3=add_fc(im_fc2,256,train_phase,'im_embed_3')
    im_fc2 = fully_connected(im_fc1, embed_dim, activation_fn=None,
                             scope = 'im_embed_2')
    i_embed = tf.nn.l2_normalize(im_fc2, 1, epsilon=1e-10)

    #im_fc3 = fully_connected(i_embed, 30, activation_fn=None,
    #                         scope = 'im_embed_3')
    #attr1 = tf.nn.sigmoid(im_fc3,name = 'attr_rec1')
    # Text branch.
    #print("type",sent_feats)
    sent_f=tf.to_int32(sent_feats,name='ToInt32')
    sent_f=tf.one_hot(sent_f,2)
    sent_fc0 = add_fc(sent_f, 4, train_phase,'sent_embed_0')
    #print("sent_one_hot",sent_f)
    #sent_fc0 = add_fc(sent_f,10,train_phase,'sent_embed_0')
    sent_fc0 = tf.layers.flatten(sent_fc0)
    #print("sent_fc_0",sent_fc0)
    #sent_fc1 = add_fc(sent_fc0, 128, train_phase,'sent_embed_1')
    #print("sent_fc1",sent_fc1)
    sent_fc2 = add_fc(sent_fc0, 256, train_phase,'sent_embed_2')
    #sent_fc3 = add_fc(sent_fc2,512 , train_phase,'sent_embed_3')
    #sent_fc4 = add_fc(sent_fc2,1024 , train_phase,'sent_embed_4')
    sent_fc3 = fully_connected(sent_fc2, embed_dim, activation_fn=None,
                               scope = 'sent_embed_3')
    s_embed = tf.nn.l2_normalize(sent_fc3, 1, epsilon=1e-10)
    #sent_fc3=im_fc3 = fully_connected(s_embed, 30, activation_fn=None,
    #                         scope = 'sent_embed_3')
    #attr2 = tf.nn.sigmoid(sent_fc3,name = 'attr_rec2')
    #attr2 = fully_connected(sent_fc2,30,activation_fn=None,scope = 'attr_rec2')
    return i_embed, s_embed


def setup_train_model(im_feats, sent_feats,im_feats_age,sent_feats_age, train_phase, im_labels, args):
    # im_feats b x image_feature_dim
    # sent_feats 5b x sent_feature_dim
    # train_phase bool (Should be True.)
    # im_labels 5b x 
    i_embed, s_embed= embedding_model(im_feats, sent_feats, train_phase, im_labels)
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        i_embed_age, s_embed_age= embedding_model(im_feats_age, sent_feats_age, train_phase, im_labels)
    #attr2 = tf.reshape(tf.tile(attr2, [1, 2]), [attr1.shape[0], -1])
    #sent_feats_t = tf.reshape(tf.tile(sent_feats, [1, 2]), [attr1.shape[0], -1])
    #attr_loss_1=tf.nn.sigmoid_cross_entropy_with_logits(logits=attr1,labels=sent_feats_t)
    #attr_loss_1=tf.reduce_mean(attr_loss_1)
    #attr_loss_2=tf.nn.sigmoid_cross_entropy_with_logits(logits=attr2,labels=sent_feats_t)
    #attr_loss_2=tf.reduce_mean(attr_loss_2)
    loss = embedding_loss(i_embed, s_embed, im_labels, args)
    loss_age = embedding_loss(i_embed_age, s_embed_age, im_labels, args)
    return 0.1*loss_age+loss

   
def setup_eval_model(im_feats, sent_feats, train_phase, im_labels):
    # im_feats b x image_feature_dim
    # sent_feats 5b x sent_feature_dim
    # train_phase bool (Should be False.)
    # im_labels 5b x b
    i_embed, s_embed = embedding_model(im_feats, sent_feats, train_phase, im_labels)
    #sent_feats_temp = tf.reshape(tf.tile(sent_feats, [1, 2]), [attr1.shape[0], -1])
    recall = recall_k(i_embed, s_embed,im_labels, ks=tf.convert_to_tensor([1,5,10]))
    return recall


def setup_sent_eval_model(im_feats, sent_feats, train_phase, im_labels, args):
    # im_feats b x image_feature_dim
    # sent_feats 5b x sent_feature_dim
    # train_phase bool (Should be False.)
    # im_labels 5b x b
    _, s_embed = embedding_model(im_feats, sent_feats, train_phase, im_labels)
    # Create 5b x 5b sentence labels, wherthe 5 x 5 blocks along the diagonal
    num_sent = args.batch_size * args.sample_size
    sent_labels = tf.reshape(tf.tile(tf.transpose(im_labels),
                                     [1, args.sample_size]), [num_sent, num_sent])
    sent_labels = tf.logical_and(sent_labels, ~tf.eye(num_sent, dtype=tf.bool))
    # For topk, query k+1 since top1 is always the sentence itself, with dist 0.
    recall = recall_k(s_embed, s_embed, sent_labels, ks=tf.convert_to_tensor([2,6,11]))[:3]
    return recall
