 CalLISI=function(emb,meta){
     lisi_index <- lisi::compute_lisi(emb, meta, c('celltype', 'BATCH'))
     clisi = median(lisi_index$celltype)
     ilisi = median(lisi_index$BATCH)
     return(c(clisi,ilisi))
 }
    