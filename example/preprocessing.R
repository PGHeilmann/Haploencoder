library(sommer)

data("DT_technow")

Md_technow[1:10,1:10]
Mf_technow[1:10,1:10]

# Missing values
sum(is.na(Md_technow));sum(is.na(Mf_technow))

# Load the map
# This file is supposed to be "genetics.114.165860-8.xls" from Technow et al. (2014), file S3
ref <- read.table("genetics.114.165860-8.xls", sep = ",", header = T)
ref[1:10,]

# Formatting of marker names is different in marker file and map file
ref$marker <- gsub(pattern = "-", replacement = ".", x = ref$marker)

all(colnames(Md_technow) %in% ref$marker)
all(colnames(Mf_technow) %in% ref$marker)

# We prepare the data for use in SelectionTools (ST). ST treats 0 as missing.
Md_technow[Md_technow == 0] <- 2
Mf_technow[Mf_technow == 0] <- 2

rownames(Md_technow) <- paste0("D", rownames(Md_technow))
rownames(Mf_technow) <- paste0("F", rownames(Mf_technow))

full.df <- rbind(Md_technow, Mf_technow)

# Save the data
write.table(ref, "st-map.txt", sep = "\t", dec = ",", row.names = F, quote = F)
write.table(t(Md_technow), "dent.txt", sep = " ", dec = ".")
write.table(t(Mf_technow), "flint.txt", sep = " ", dec = ".")
write.table(t(full.df), "flintdent.txt", sep = " ", dec = ".")

###############################
#       Data filtering        #
###############################

library(SelectionTools)

st.read.marker.data("flintdent.txt", format = "m")

st.restrict.marker.data(NoAll.MAX = 2)
st.restrict.marker.data(ExHet.MIN = .05)

gs.build.Z()
Z <- gs.get.Z()
x <- st.marker.data.statistics()

rownames(Z) <- x$individual.list$Name
colnames(Z) <- x$marker.list$Name

Z[1:10,1:10]
write.table(Z, "dentflint_filtered.txt")

###############################
#         Haploblocks         #
###############################

st.read.map(filename = "st-map.txt", skip = 1, format = "mcp")

ld <- st.calc.ld(ld.measure="r2")

h <- st.def.hblocks ( ld.threshold = 0.7, tolerance = 1,
                      ld.criterion = "flanking",
                      data.set="default" )

write.table(h, "LD07FL.csv", sep = "\t")
