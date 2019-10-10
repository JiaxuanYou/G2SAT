from args import *
from model import *
from train import *
from data import *
from utils import *
from tensorboardX import SummaryWriter


### args
args = make_args()
print(args)
np.random.seed(123)

### set up tensorboard writer
args.name = '{}_{}_{}_pre{}_drop{}_yield{}_{}'.format(
    args.model, args.layer_num, args.hidden_dim, args.feature_pre, args.dropout, args.yield_prob, args.comment)
writer_train = SummaryWriter(comment=args.name+'train')
writer_test = SummaryWriter(comment=args.name+'test')
args.graphs_save_path = 'graphs/'

### set up gpu
if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
else:
    print('Using CPU')
device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')

### load data
graphs_train, nodes_par1s_train, nodes_par2s_train = load_graphs_lcg(data_dir='dataset/train_set/', stats_dir='dataset/')
# draw_graph_list(graphs_train, row=4, col=4, fname='fig/train')
node_nums = [graph.number_of_nodes() for graph in graphs_train]
edge_nums = [graph.number_of_edges() for graph in graphs_train]
print('Num {}, Node {} {} {}, Edge {} {} {}'.format(
    len(graphs_train),min(node_nums),max(node_nums),sum(node_nums)/len(node_nums),min(edge_nums),max(edge_nums),sum(edge_nums)/len(edge_nums)))

dataset_train = Dataset_sat(graphs_train,nodes_par1s_train,nodes_par2s_train,epoch_len=5000, yield_prob=args.yield_prob, speedup=True)
dataset_test = Dataset_sat(graphs_train,nodes_par1s_train,nodes_par2s_train,epoch_len=1000, yield_prob=args.yield_prob, speedup=False)


loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.worker_num)
loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=args.worker_num)


input_dim = 3 # 3 node types
model = locals()[args.model](input_dim=input_dim, feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim, output_dim=args.output_dim,
                feature_pre=args.feature_pre, layer_num=args.layer_num, dropout=args.dropout).to(device)
if args.load_model:
    model_fname = 'model/'+args.name+str(args.epoch_load)
    model.load_state_dict(torch.load(model_fname))
    model.eval()


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)


train(args, loader_train, loader_test, model, optimizer, writer_train, writer_test, device)



