import numpy as np from sklearn.datasets import load_irisfrom sklearn.model_selection import train_test_splitfrom sklearn.metrics import accuracy_scoredef entropy(Y):    _,a = np.unique(Y,return_counts=True)    a = a/a.sum()    e = (np.log2(a) * a).sum()    return -edef split(X,Y,max_features=None):    assert len(X)>1    N,M = X.shape    point,e_child = (None,None),9999999    if max_features is None:        feature_idx = range(M)    else:        feature_idx = np.random.choice(M, max_features, replace=False)            for col in feature_idx:         unique_vals = np.unique(X[:,col])        for i in range(1,len(unique_vals)):            v = (unique_vals[i]+unique_vals[i-1])/2            idx = X[:,col]<=v            ea = entropy(Y[idx])            eb = entropy(Y[~idx])                        la = idx.sum()            e = (la*ea+(N-la)*eb) / N            if e<=e_child:                point,e_child = (v,col),e         assert point[0] is not None and point[1] is not None            return pointclass Tree():        defaults = {'max_features':None,'max_depth':None,'min_samples_split':2}        def __init__(self,cur_depth = 0,**kargs):        assert all(k in Tree.defaults.keys() for k in kargs),"Unrecognized keyword parameter"        self.args = {k:kargs[k] if k in kargs else Tree.defaults[k] for k in Tree.defaults.keys()}        self.cur_depth = cur_depth                        self.split_col = None        self.split_val = None        self.left = None        self.right = None                       self.label = None                      def fit(self,X,Y):        assert np.issubdtype(Y.dtype,np.integer),"Target column should be integer"                if len(X)<self.args['min_samples_split'] or self.cur_depth == self.args['max_depth']:            self.label = np.bincount(Y).argmax()            return self                if len(np.unique(Y))==1:            self.label = Y[0]            return self                self.split_val,self.split_col = split(X,Y,self.args['max_features'])                idx = X[:,self.split_col] <= self.split_val        self.left = Tree(cur_depth=self.cur_depth+1, **self.args).fit(X[idx],Y[idx])        self.right = Tree(cur_depth=self.cur_depth+1,**self.args).fit(X[~idx],Y[~idx])                return self            def predict(self,X):                if self.label is not None:            assert self.left is None and self.right is None            return self.label                ans = np.zeros(len(X))        idx = X[:,self.split_col] <= self.split_val        ans[idx] = self.left.predict(X[idx])         ans[~idx] = self.right.predict(X[~idx])        return ans        def traverse(self,space=0):        if self.label is not None:            print( f"{' '*space}LEAF {self.label}")            return                print(f"{' '*space}{self.split_col},{self.split_val}")        self.left.traverse(space+4)        self.right.traverse(space+4)    if __name__=='__main__':    Xx,Yy = load_iris(return_X_y=True)    x1,x2,y1,y2 = train_test_split(Xx,Yy,test_size=.2)    tree  = Tree()    tree = tree.fit(x1,y1)    yp = np.array(tree.predict(x2))    print(accuracy_score(y2,yp))