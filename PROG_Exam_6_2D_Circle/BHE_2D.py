"""
@author: Farinaz Mostajeran
"""



import tensorflow as tf
import numpy as np

import scipy.io
import scipy.io as sio
from scipy.interpolate import griddata
import time

np.random.seed(1234)
tf.set_random_seed(1234)

class DeepBHCP:
    # Initialize the class
    def __init__(self, x, y, t, u,  x_b, y_b, t_b, u_b, x_T, y_T, t_T, u_T,layers):
        
        X = np.concatenate([x, y, t], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.t = X[:,2:3]
        self.u = u  # NOT USED
        
        X_b = np.concatenate([x_b, y_b, t_b], 1)
        self.X_b = X_b
        self.x_b = X_b[:,0:1]
        self.y_b = X_b[:,1:2]
        self.t_b = X_b[:,2:3]
        self.u_b = u_b
        
        X_T = np.concatenate([x_T, y_T, t_T], 1)
        self.X_T = X_T
        self.x_T = X_T[:,0:1]
        self.y_T = X_T[:,1:2]
        self.t_T = X_T[:,2:3]
        self.u_T = u_T
        
        self.layers = layers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)        
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        #self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        
        self.xb_tf = tf.placeholder(tf.float32, shape=[None, self.x_b.shape[1]])
        self.yb_tf = tf.placeholder(tf.float32, shape=[None, self.y_b.shape[1]])
        self.tb_tf = tf.placeholder(tf.float32, shape=[None, self.t_b.shape[1]])
        self.ub_tf = tf.placeholder(tf.float32, shape=[None, self.u_b.shape[1]])
        
        self.xT_tf = tf.placeholder(tf.float32, shape=[None, self.x_T.shape[1]])
        self.yT_tf = tf.placeholder(tf.float32, shape=[None, self.y_T.shape[1]])
        self.tT_tf = tf.placeholder(tf.float32, shape=[None, self.t_T.shape[1]])
        self.uT_tf = tf.placeholder(tf.float32, shape=[None, self.u_T.shape[1]])
        
        self.u_pred, self.f_pred = self.net_NS(self.x_tf, self.y_tf, self.t_tf)
        self.ub_pred = self.net_u(self.xb_tf, self.yb_tf, self.tb_tf)
        self.uT_pred = self.net_u(self.xT_tf, self.yT_tf, self.tT_tf)
        
        
        self.loss = tf.reduce_mean(tf.square(self.ub_tf - self.ub_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred)) + \
                    tf.reduce_mean(tf.square(self.uT_tf - self.uT_pred))
                    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_u(self, x, y, t):
        u = self.neural_net(tf.concat([x, y, t],1), self.weights, self.biases)
        return u
        
    def net_NS(self, x, y, t):
        
        u = self.net_u(x, y, t)
        
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        
        F = (y*tf.sin(np.pi*x) + x*tf.cos(np.pi*y)) * (tf.sin(t)-np.pi*np.pi*tf.cos(t))

        f_u = u_t + F - (u_xx + u_yy)
        
        return u, f_u
    
    def callback(self, loss):
        print('Loss:', loss)
      
    def train(self, nIter): 

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t,
                   self.xb_tf: self.x_b, self.yb_tf: self.y_b, self.tb_tf: self.t_b,
                   self.ub_tf: self.u_b,
                   self.xT_tf: self.x_T, self.yT_tf: self.y_T, self.tT_tf: self.t_T,
                   self.uT_tf: self.u_T}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % (it, loss_value, elapsed))
                start_time = time.time()
            
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)
            
    
    def predict(self, x_star, y_star, t_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)

        return u_star
        
        
if __name__ == "__main__": 
       
    noise = 0.00
      
    N_train = 4000
    N_b = 200
    N_T = 200
    nIter = 0
    
    layers = [3, 20, 20, 20, 1]
    
    Name_BHCP = 'BHE_2Dcircle_T1_noise000_1.mat'
    
    # Load Data
    data = scipy.io.loadmat('../Data/BHE2D_6_T1_circleX200.mat')
           
    U_star = data['U_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2
    N = X_star.shape[0] # n1 x n2
    T = t_star.shape[0]
    
    #U_sol = data['U_sol'] # n1 x n2 x T
    X_bound_1 = data['X_bound_1'] # MX2
    U_bound_1 = data['U_bound_1'] #MX2
    
    
    U_bound = data['U_bound'] # (2x(n1+n2)xT) x 1
    X_bound = data['X_bound'] # (2x(n1+n2)xT) x 3
    Nb = X_bound.shape[0] #  = 2x(n1+n2)xT
    
    
    # Rearrange all Data 
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T
    UU = U_star # N x T
    
    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1
    u = UU.flatten()[:,None] # NT x 1
    
    # Rearrange bound Data 
    xb = X_bound[:,0:1].flatten()[:,None] # Nb x 1
    yb = X_bound[:,1:2].flatten()[:,None] # Nb x 1
    tb = X_bound[:,2:3].flatten()[:,None] # Nb x 1
    ub = U_bound.flatten()[:,None] # Nb x 1
    
    # Rearrange final Data 
    xT = X_star[:,0:1].flatten()[:,None] # 
    yT = X_star[:,1:2].flatten()[:,None] #  
    tT = TT[:,-1:].flatten()[:,None] #  
    uT = U_star[:,-1:].flatten()[:,None] #  
        
    ########################  Data #######################################
    # Training Data from all   
    idx = np.random.choice(N*T, N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]
    
    # Training Data from bound   
    idx = np.random.choice(Nb, N_b, replace=False)
    xb_train = xb[idx,:]
    yb_train = yb[idx,:]
    tb_train = tb[idx,:]
    ub_train = ub[idx,:]
    
    # Training Data from final time 
    idx = np.random.choice(xT.shape[0], N_T, replace=False)
    xT_train = xT[idx,:]
    yT_train = yT[idx,:]
    tT_train = tT[idx,:]
    uT_train = uT[idx,:]

    
    u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])  
    ub_train = ub_train + noise*np.std(ub_train)*np.random.randn(ub_train.shape[0], ub_train.shape[1])
    uT_train = uT_train + noise*np.std(uT_train)*np.random.randn(uT_train.shape[0], uT_train.shape[1])
    
    
    # Training
    model = DeepBHCP(x_train, y_train, t_train, u_train, xb_train, yb_train, tb_train, ub_train, xT_train, yT_train, tT_train, uT_train, layers)
    start_time_1 = time.time() 
    #model.train(200000)
    model.train(nIter)
    elapsed = time.time() - start_time_1                
    print('Training time: %.4f' % (elapsed))
    
    # Test Data
    snap = np.array([0])
    """
    x_star = X_star[:,0:1]
    y_star = X_star[:,1:2]
    t_star = TT[:,snap]
    u_star = U_star[:, snap]
    """
    x_star = np.hstack((X_star[:,0:1].T,X_bound_1[:,0:1].T)).T
    y_star = np.hstack((X_star[:,1:2].T, X_bound_1[:,1:2].T)).T
    t_star = np.hstack((TT[:,snap].T,np.array(t_star[snap]*np.ones(U_bound_1.shape)).T)).T
    u_star = np.hstack((U_star[:, snap].T, U_bound_1.T)).T
    #u_star = U_star[:, snap]
    
    # Prediction
    u_pred = model.predict(x_star, y_star, t_star)
    
    # Error
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

    
    print('Error u0: %e' % (error_u))               
    
    X_star = np.vstack((x_star.T,y_star.T)).T
    
    
    # Predict for plotting
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
    U_exact = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    
    
    
    sio.savemat(Name_BHCP, {'U_pred_0':u_pred, 'U_exact':u_star, 
                                              'N_b':N_b, 'N_b':N_b, 'N_T':N_T, 'N_train':N_train, 
                                              'Layers':layers, 'u_train':uT, 
                                              'x_star':x_star, 'y_star':y_star,
                                              'ErrorU':error_u,'run_time':elapsed})
    model.sess.close()
    