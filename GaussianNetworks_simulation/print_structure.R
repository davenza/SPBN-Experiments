require(bnlearn)


load("ecoli70.rda")
ecoli70 <- bn
load("magic-niab.rda")
magic_niab <- bn
load("magic-irri.rda")
magic_irri <- bn
load("arth150.rda")
arth150 <- bn

print_edges <- function(bn.fit) {
  arcs <- bn.net(bn.fit)$arcs
  cat(paste0("[(\"", arcs[1,1], "\", \"", arcs[1,2], "\")"))
  for (row in 2:nrow(arcs)) {
    cat(paste0(", (\"", arcs[row,1], "\", \"", arcs[row, 2], "\")"))
  }
  
  cat("]\n")
  print("")
}

print_cpds <- function(bn.fit) {
  
  for (node in names(bn.fit)) {
    name <- paste0("\"", bn.fit[[node]]$node, "\"")
    
    n_parents <- length(bn.fit[[node]]$parents)
    if (n_parents == 0) {
      beta <- paste0("[", bn.fit[[node]]$coefficients[["(Intercept)"]], "]")
      parents <- "[]"
    }
    else {
      first_parent <- bn.fit[[node]]$parents[1] 
      beta <- paste0("[", bn.fit[[node]]$coefficients[["(Intercept)"]], ", ", bn.fit[[node]]$coefficients[[first_parent]])
      parents <- paste0("[\"", first_parent, "\"")
      
      if (n_parents > 1) {
        for (parent in bn.fit[[node]]$parents[2:n_parents]) {
          beta <- paste0(beta, ", ", bn.fit[[node]]$coefficients[[parent]])
          parents <- paste0(parents, ", \"", parent, "\"")
        }
      }

      beta <- paste0(beta, "]")
      parents <- paste0(parents, "]")
      
      variance <- bn.fit[[node]]$sd**2
    }

    cat(paste0("n", node, "_cpd = LinearGaussianCPD(",  name, ", ", parents, ", ", beta, ", ", variance, ")\n"))
  }
  
  add <- paste0("add_cpds(n", names(bn.fit)[1], "_cpd")
  n_nodes <- length(names(bn.fit))
  for (node in names(bn.fit)[2:n_nodes]) {
    add <- paste0(add, ", n", node, "_cpd")
  }
  add <- paste0(add, ")")
  cat(paste0(add, "\n"))
}


print("ECOLI70")
print("=======================")
print("")
print_edges(ecoli70)
print_cpds(ecoli70)



print("")
print("")

print("MAGIC-NIAB")
print("=======================")
print("")

print_edges(magic_niab)
print_cpds(magic_niab)


print("")
print("")

print("MAGIC-IRRI")
print("=======================")
print("")

print_edges(magic_irri)
print_cpds(magic_irri)


print("")
print("")

print("ARTH150")
print("=======================")
print("")

print_edges(arth150)
print_cpds(arth150)
