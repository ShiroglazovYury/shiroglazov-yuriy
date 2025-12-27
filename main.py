import pandas as pd, numpy as np, pickle, warnings
from catboost import CatBoostRegressor, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

BUNDLE="bundle.pkl"
MFILES=dict(mid_w="m_mid_w.cbm",mid_c="m_mid_c.cbm",lw_w="m_lw_w.cbm",lw_c="m_lw_c.cbm")
TR_PATH="data/train.csv"
TE_PATH="data/test.csv"
OUT_PATH="results/submission.csv"

b=pickle.load(open(BUNDLE,"rb"))
RS,PARAMS,base,knn_feats,cat_all,cat_cold,warm_cols,cold_cols,prod2cl,cat2cl,CAL,ref=b["RS"],b["PARAMS"],b["base"],b["knn_feats"],b["cat_all"],b["cat_cold"],b["warm_cols"],b["cold_cols"],b["prod2cl"],b["cat2cl"],b["CAL"],b["ref"]

add_feats=lambda d:(d.assign(
    n_stores_log=np.log1p(d["n_stores"].clip(lower=0)),precpt_log=np.log1p(d["precpt"].clip(lower=0)),
    temp2=d["avg_temperature"]**2,hum2=d["avg_humidity"]**2,wind2=d["avg_wind_level"]**2,
    temp_x_hum=d["avg_temperature"]*d["avg_humidity"],temp_x_prec=d["avg_temperature"]*np.log1p(d["precpt"].clip(lower=0)),
    hum_x_wind=d["avg_humidity"]*d["avg_wind_level"],temp_x_wind=d["avg_temperature"]*d["avg_wind_level"],
    prec_x_hum=np.log1p(d["precpt"].clip(lower=0))*d["avg_humidity"],is_weekend=d["dow"].isin([5,6]).astype(int),
    is_month_start=(d["day_of_month"]<=3).astype(int),is_month_end=(d["day_of_month"]>=28).astype(int),
    dow_sin=np.sin(2*np.pi*d["dow"]/7),dow_cos=np.cos(2*np.pi*d["dow"]/7),
    month_sin=np.sin(2*np.pi*(d["month"]-1)/12),month_cos=np.cos(2*np.pi*(d["month"]-1)/12),
    woy_sin=np.sin(2*np.pi*(d["week_of_year"]-1)/52),woy_cos=np.cos(2*np.pi*(d["week_of_year"]-1)/52),
    dom_sin=np.sin(2*np.pi*(d["day_of_month"]-1)/31),dom_cos=np.cos(2*np.pi*(d["day_of_month"]-1)/31),
    act_x_dow=d["activity_flag"]*d["dow"],hol_x_dow=d["holiday_flag"]*d["dow"],act_x_woy=d["activity_flag"]*d["week_of_year"],
    temp_x_act=d["avg_temperature"]*d["activity_flag"],prec_x_act=np.log1p(d["precpt"].clip(lower=0))*d["activity_flag"]
))

def build_feats(df,ref):
    df=df.copy(); ref=ref.copy().sort_values("dt")
    df["sc_dow_act"]=df["second_category_id"]+"|"+df["dow"].astype(str)+"|"+df["activity_flag"].astype(str)
    df["sc_act"]=df["second_category_id"]+"|"+df["activity_flag"].astype(str)
    df["mg_fc_sc"]=df["management_group_id"]+"|"+df["first_category_id"]+"|"+df["second_category_id"]
    df["fc_sc"]=df["first_category_id"]+"|"+df["second_category_id"]
    prod=ref.groupby("product_id").agg(prod_days=("mid","size"),prod_mid_mean=("mid","mean"),prod_mid_med=("mid","median"),prod_lw_mean=("logw","mean"),prod_lw_med=("logw","median"))
    seg=ref.groupby("second_category_id").agg(seg_rows=("mid","size"),seg_prods=("product_id","nunique"),seg_mid_mean=("mid","mean"),seg_mid_med=("mid","median"),seg_lw_mean=("logw","mean"),seg_lw_med=("logw","median"))
    seg_ctx=ref.groupby(["second_category_id","dow","activity_flag"]).agg(seg_ctx_rows=("mid","size"),seg_mid_ctx_mean=("mid","mean"),seg_mid_ctx_med=("mid","median"),seg_lw_ctx_mean=("logw","mean"),seg_lw_ctx_med=("logw","median"))
    last=ref.groupby("product_id").tail(1).set_index("product_id")[["mid","logw"]].rename(columns={"mid":"mid_last","logw":"lw_last"})
    df=df.join(prod,on="product_id").join(seg,on="second_category_id").join(seg_ctx,on=["second_category_id","dow","activity_flag"]).join(last,on="product_id")
    df["prod_days"]=df["prod_days"].fillna(0); df["is_new_product"]=(df["prod_days"]==0).astype(int)
    gmid,glw=float(ref["mid"].mean()),float(ref["logw"].mean())
    for c in ["prod_mid_mean","prod_mid_med","seg_mid_mean","seg_mid_med","seg_mid_ctx_mean","seg_mid_ctx_med","mid_last"]: df[c]=df[c].fillna(gmid)
    for c in ["prod_lw_mean","prod_lw_med","seg_lw_mean","seg_lw_med","seg_lw_ctx_mean","seg_lw_ctx_med","lw_last"]: df[c]=df[c].fillna(glw)
    for c in ["seg_ctx_rows","seg_rows","seg_prods"]: df[c]=df[c].fillna(0)
    return df

def knn_hier(ref,qry):
    qry=qry.reset_index(drop=True); out=np.zeros((len(qry),3),float); out[:,0]=ref.mid.mean(); out[:,1]=ref.logw.mean()
    q=qry.copy(); r=ref.copy()
    q["key1"]=q["second_category_id"].astype(str)+"|"+q["dow"].astype(str)+"|"+q["activity_flag"].astype(str)
    q["key2"]=q["second_category_id"].astype(str)+"|"+q["activity_flag"].astype(str)
    q["key3"]=q["second_category_id"].astype(str)
    r["key1"]=r["second_category_id"].astype(str)+"|"+r["dow"].astype(str)+"|"+r["activity_flag"].astype(str)
    r["key2"]=r["second_category_id"].astype(str)+"|"+r["activity_flag"].astype(str)
    r["key3"]=r["second_category_id"].astype(str)
    Xr=r[knn_feats].fillna(0).to_numpy()
    sc=StandardScaler(); Xs=sc.fit_transform(Xr)
    p=PCA(n_components=PARAMS["pca"],random_state=RS).fit(Xs)
    for keycol in ["key1","key2","key3"]:
        for key,pos in q.groupby(keycol).groups.items():
            pos=list(pos)
            if np.all(out[pos,2]>0): continue
            rr=r[r[keycol]==key]
            if len(rr)<5:
                if len(rr)>0:
                    m,lw=rr.mid.mean(),rr.logw.mean()
                    m2=q.loc[pos,"seg_mid_ctx_med"].to_numpy(); lw2=q.loc[pos,"seg_lw_ctx_med"].to_numpy()
                    use=(out[pos,2]==0)
                    out[pos,0]=np.where(use,0.5*m+0.5*m2,out[pos,0])
                    out[pos,1]=np.where(use,0.5*lw+0.5*lw2,out[pos,1])
                    out[pos,2]=np.where(use,len(rr),out[pos,2])
                continue
            Xrr=p.transform(sc.transform(rr[knn_feats].fillna(0).to_numpy()))
            Xqq=p.transform(sc.transform(q.loc[pos,knn_feats].fillna(0).to_numpy()))
            nn=NearestNeighbors(n_neighbors=min(PARAMS["k"],len(rr)),metric="euclidean",n_jobs=8).fit(Xrr)
            d,ii=nn.kneighbors(Xqq); w=(1/(d+1e-6)); w=w/w.sum(1,keepdims=True)
            Y=rr[["mid","logw"]].to_numpy()[ii]; pr=(Y*w[...,None]).sum(1)
            ne=(w.sum(1)**2/(w**2).sum(1)); lam=np.clip(ne/10,0,1); use=(out[pos,2]==0)
            m2=q.loc[pos,"seg_mid_ctx_med"].to_numpy(); lw2=q.loc[pos,"seg_lw_ctx_med"].to_numpy()
            out[pos,0]=np.where(use,lam*pr[:,0]+(1-lam)*m2,out[pos,0])
            out[pos,1]=np.where(use,lam*pr[:,1]+(1-lam)*lw2,out[pos,1])
            out[pos,2]=np.where(use,ne,out[pos,2])
    return out

loadm=lambda p:(lambda m:(m.load_model(p),m)[1])(CatBoostRegressor())
pred=lambda m,cat_cols,X:(lambda idx: np.array(m.predict(Pool(X,cat_features=idx))))([X.columns.get_loc(c) for c in cat_cols])
gate=lambda x,a,b: 1/(1+np.exp(-(a+b*np.log1p(x))))
blend=lambda xw,xc,xk,w,b: w*xw+(1-w)*(b*xc+(1-b)*xk)
to_interval=lambda mid,lw:(lambda w: np.c_[np.maximum(mid-w/2,0), np.maximum(mid+w/2,0)])(np.expm1(np.maximum(lw,0)))

def backoff(df,d3,d2,d1,g):
    sc=df["second_category_id"].astype(str).to_numpy(); dow=df["dow"].astype(int).to_numpy(); act=df["activity_flag"].astype(int).to_numpy()
    return np.array([d3.get((sc[i],dow[i],act[i]), d2.get((sc[i],dow[i]), d1.get(sc[i], g))) for i in range(len(df))],float)

def apply_cal(df_like,mid,lw,prod_days):
    DPM,GPM,DPL,GPL,D3M,D2M,D1M,GM,D3W,D2W,D1W,GW=CAL
    df=pd.DataFrame({"product_id":df_like["product_id"].astype(str),"second_category_id":df_like["second_category_id"].astype(str),"dow":df_like["dow"].astype(int),"activity_flag":df_like["activity_flag"].astype(int)})
    isw=prod_days>0
    addm=np.zeros(len(df)); addw=np.zeros(len(df))
    if isw.any():
        pid=df.loc[isw,"product_id"].to_numpy()
        addm[isw]=np.array([DPM.get(p,GPM) for p in pid]); addw[isw]=np.array([DPL.get(p,GPL) for p in pid])
    if (~isw).any():
        cd=df.loc[~isw]; addm[~isw]=backoff(cd,D3M,D2M,D1M,GM); addw[~isw]=backoff(cd,D3W,D2W,D1W,GW)
    return mid+addm, lw+addw

m_mid_w,m_mid_c,m_lw_w,m_lw_c=(loadm(MFILES["mid_w"]),loadm(MFILES["mid_c"]),loadm(MFILES["lw_w"]),loadm(MFILES["lw_c"]))

te=pd.read_csv(TE_PATH); te["dt"]=pd.to_datetime(te["dt"])
for c in base: te[c]=te[c].astype(str)
te=add_feats(te)
te["product_cluster_id"]=te["product_id"].astype(str).map(prod2cl).fillna(-1).astype(int)
te.loc[te["product_cluster_id"]==-1,"product_cluster_id"]=te.loc[te["product_cluster_id"]==-1,"second_category_id"].map(cat2cl).fillna(-1).astype(int)
te["product_cluster_id"]=te["product_cluster_id"].astype(str)

ref=ref.copy()
tf=build_feats(te,ref)
kn=knn_hier(ref,tf)

mid=blend(pred(m_mid_w,cat_all,tf[warm_cols]),pred(m_mid_c,cat_cold,tf[cold_cols]),kn[:,0],gate(tf["prod_days"].values,PARAMS["aM"],PARAMS["bM"]),PARAMS["betaM"])
lw =blend(pred(m_lw_w,cat_all,tf[warm_cols]),pred(m_lw_c,cat_cold,tf[cold_cols]),kn[:,1],gate(tf["prod_days"].values,PARAMS["aW"],PARAMS["bW"]),PARAMS["betaW"])
lw =lw+np.where(tf["prod_days"].values>0,PARAMS["shw"],PARAMS["shc"]+PARAMS["ec"])
mid,lw=apply_cal(te,mid,lw,tf["prod_days"].values)

pr=to_interval(mid,lw); pr[:,1]=np.maximum(pr[:,1],pr[:,0])
pd.DataFrame({"row_id":te["row_id"],"price_p05":pr[:,0],"price_p95":pr[:,1]}).to_csv(OUT_PATH,index=False)
print("saved",OUT_PATH)