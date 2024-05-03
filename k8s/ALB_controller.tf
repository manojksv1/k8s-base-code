# # Install the AWS Load Balancer Controller using Helm
# resource "helm_release" "aws_load_balancer_controller" {
#   name             = "aws-load-balancer-controller"
#   repository       = "https://aws.github.io/eks-charts"
#   chart            = "aws-load-balancer-controller"
#   namespace        = "kube-system"
#   create_namespace = true
#   version          = "1.7.2"

#   set {
#     name  = "clusterName"
#     value = aws_eks_cluster.eks_cluster.name
#   }
#   set {
#     name  = "serviceAccount.create"
#     value = "false"
#   }
#   set {
#     name  = "serviceAccount.name"
#     value = "aws-load-balancer-controller"
#   }
#   # depends_on = [ service_accounts ]
# }

resource "helm_release" "aws-load-balancer-controller" {
  name = "aws-load-balancer-controller"

  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-load-balancer-controller"
  namespace  = "kube-system"
  version    = "1.7.2"

  set {
    name  = "clusterName"
    value = aws_eks_cluster.eks_cluster.name
  }

  set {
    name  = "image.tag"
    value = "v2.7.2"
  }

  set {
    name  = "serviceAccount.name"
    value = "aws-load-balancer-controller"
  }

  set {
    name  = "serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = aws_iam_role.aws_load_balancer_controller.arn
  }

  depends_on = [
    aws_eks_node_group.private-nodes,
    aws_iam_role_policy_attachment.aws_load_balancer_controller_attach
  ]
}