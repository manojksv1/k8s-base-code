# Helm repositories are no longer defined as data sources
variable "stable_repo_url" {
  type    = string
  default = "https://charts.helm.sh/stable"
}

variable "prometheus_community_repo_url" {
  type    = string
  default = "https://prometheus-community.github.io/helm-charts"
}

# Install Prometheus using Helm chart
resource "helm_release" "prometheus" {
  name             = "prometheus"
  namespace        = "prometheus"
  create_namespace = true
  repository       = var.prometheus_community_repo_url
  chart            = "kube-prometheus-stack"
  depends_on       = [aws_eks_cluster.eks_cluster, aws_eks_node_group.private-nodes]
}