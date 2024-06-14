Remove-Item metrics.csv -ErrorAction Ignore

while ($true) {
    $pods = microk8s kubectl get pods --all-namespaces -o json | ConvertFrom-Json
    $nodeCounts = @{}

    foreach ($pod in $pods.items) {
        if ($pod.status.phase -eq 'Running' -and $pod.metadata.namespace -ne 'kube-system') {
            $node = $pod.spec.nodeName
            if ($nodeCounts.ContainsKey($node)) {
                $nodeCounts[$node]++
            } else {
                $nodeCounts[$node] = 1
            }
        }
    }

    microk8s kubectl top nodes | ForEach-Object {
        if ($_ -match '(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)') {
            $node = $matches[1]
            $podCount = $nodeCounts[$node]
            "$node,$($matches[2]),$($matches[3]),$($matches[4]),$($matches[5]),$podCount" >> metrics.csv
        }
    }

    Start-Sleep -Seconds 60
}