data "aws_eks_cluster" "eks" {
  name = aws_eks_cluster.eks_cluster.name

  depends_on = [aws_eks_cluster.eks_cluster]
}

data "aws_eks_cluster_auth" "eks" {
  name = aws_eks_cluster.eks_cluster.name

  depends_on = [aws_eks_cluster.eks_cluster]
}

provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.eks.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.eks.certificate_authority[0].data)
    token                  = data.aws_eks_cluster_auth.eks.token
  }
}